from typing import Callable, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_big_bird import BigBirdConfig
class FlaxBigBirdBlockSparseAttention(nn.Module):
    config: BigBirdConfig
    block_sparse_seed: int = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.query = nn.Dense(self.config.hidden_size, dtype=self.dtype, use_bias=self.config.use_bias, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        self.key = nn.Dense(self.config.hidden_size, dtype=self.dtype, use_bias=self.config.use_bias, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        self.value = nn.Dense(self.config.hidden_size, dtype=self.dtype, use_bias=self.config.use_bias, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))

    @staticmethod
    def transpose_for_scores(x, n_heads, head_size):
        new_x_shape = x.shape[:-1] + (n_heads, head_size)
        x = x.reshape(*new_x_shape)
        return jnp.transpose(x, axes=(0, 2, 1, 3))

    def __call__(self, hidden_states, attention_mask, deterministic=True, output_attentions=False):
        n_heads = self.config.num_attention_heads
        head_size = self.config.hidden_size // n_heads
        blocked_encoder_mask, band_mask, from_mask, to_mask = self.create_masks_for_block_sparse_attn(attention_mask, self.config.block_size)
        query_layer = self.transpose_for_scores(self.query(hidden_states), n_heads, head_size)
        key_layer = self.transpose_for_scores(self.key(hidden_states), n_heads, head_size)
        value_layer = self.transpose_for_scores(self.value(hidden_states), n_heads, head_size)
        indices_prng_key = None
        if not deterministic:
            indices_prng_key = self.make_rng('indices')
        attn_output, attn_weights = self.bigbird_block_sparse_attention(query_layer, key_layer, value_layer, band_mask, from_mask, to_mask, blocked_encoder_mask, blocked_encoder_mask, n_heads, head_size, indices_prng_key=indices_prng_key, deterministic=deterministic, plan_from_length=None, plan_num_rand_blocks=None, output_attentions=output_attentions)
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs

    @staticmethod
    def create_masks_for_block_sparse_attn(attention_mask, block_size: int):
        batch_size, seq_length = attention_mask.shape
        if seq_length % block_size != 0:
            raise ValueError(f'Sequence length must be multiple of block size, but sequence length is {seq_length}, while block size is {block_size}.')

        def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
            """
            Create 3D attention mask from a 2D tensor mask.

            Args:
                from_blocked_mask: 2D Tensor of shape [batch_size,
                from_seq_length//from_block_size, from_block_size].
                to_blocked_mask: int32 Tensor of shape [batch_size,
                to_seq_length//to_block_size, to_block_size].

            Returns:
                float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4, from_block_size,
                3*to_block_size].
            """
            exp_blocked_to_pad = jnp.concatenate([to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]], axis=2)
            band_mask = jnp.einsum('blq,blk->blqk', from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
            band_mask = jnp.expand_dims(band_mask, 1)
            return band_mask
        blocked_encoder_mask = attention_mask.reshape(batch_size, seq_length // block_size, block_size)
        band_mask = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)
        from_mask = attention_mask.reshape(batch_size, 1, seq_length, 1)
        to_mask = attention_mask.reshape(batch_size, 1, 1, seq_length)
        return (blocked_encoder_mask, band_mask, from_mask, to_mask)

    def bigbird_block_sparse_attention(self, query_layer, key_layer, value_layer, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask, n_heads, head_size, indices_prng_key: Optional[jax.random.PRNGKey]=None, deterministic: Optional[bool]=True, plan_from_length=None, plan_num_rand_blocks=None, output_attentions=None):
        bsz, _, from_seq_len, _ = query_layer.shape
        to_seq_len = key_layer.shape[2]
        from_block_size = to_block_size = self.config.block_size
        if from_seq_len % from_block_size != 0:
            raise ValueError('Query sided sequence length must be multiple of block size')
        if to_seq_len % to_block_size != 0:
            raise ValueError('Key/Value sided sequence length must be multiple of block size')
        if from_seq_len // from_block_size != to_seq_len // to_block_size:
            raise ValueError('Error the number of blocks needs to be same!')
        n_rand_blocks = self.config.num_random_blocks
        rsqrt_d = 1 / jnp.sqrt(head_size)
        attn_mask_penalty = -10000.0
        if from_seq_len in [1024, 3072, 4096]:
            max_seqlen = self.config.max_position_embeddings
            rand_attn = [self._bigbird_block_rand_mask(max_seqlen, max_seqlen, from_block_size, to_block_size, n_rand_blocks, indices_prng_key=indices_prng_key, deterministic=deterministic, last_idx=1024)[:from_seq_len // from_block_size - 2] for _ in range(n_heads)]
        else:
            if plan_from_length is None:
                plan_from_length, plan_num_rand_blocks = self._get_rand_attn_plan(from_seq_len, from_block_size, n_rand_blocks)
            rand_attn = self._bigbird_block_rand_mask_with_head(from_seq_length=from_seq_len, to_seq_length=to_seq_len, from_block_size=from_block_size, to_block_size=to_block_size, num_heads=n_heads, plan_from_length=plan_from_length, plan_num_rand_blocks=plan_num_rand_blocks, indices_prng_key=indices_prng_key)
        rand_attn = jnp.stack(rand_attn, axis=0)
        rand_attn = jnp.broadcast_to(rand_attn, (bsz,) + rand_attn.shape)
        rand_mask = self._create_rand_mask_from_inputs(from_blocked_mask, to_blocked_mask, rand_attn, n_heads, n_rand_blocks, bsz, from_seq_len, from_block_size)
        blocked_query_matrix = query_layer.reshape(bsz, n_heads, from_seq_len // from_block_size, from_block_size, -1)
        blocked_key_matrix = key_layer.reshape(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)
        blocked_value_matrix = value_layer.reshape(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)
        shape = (bsz, n_heads, to_seq_len // to_block_size - 2, n_rand_blocks * to_block_size, -1)
        gathered_key = self.jax_gather(blocked_key_matrix, rand_attn, batch_dims=2).reshape(*shape)
        gathered_value = self.jax_gather(blocked_value_matrix, rand_attn, batch_dims=2).reshape(*shape)
        first_product = jnp.einsum('bhqd,bhkd->bhqk', blocked_query_matrix[:, :, 0], key_layer)
        first_product = first_product * rsqrt_d
        first_product += (1.0 - to_mask) * attn_mask_penalty
        first_attn_weights = jax.nn.softmax(first_product, axis=-1)
        first_context_layer = jnp.einsum('bhqk,bhkd->bhqd', first_attn_weights, value_layer)
        first_context_layer = jnp.expand_dims(first_context_layer, 2)
        second_key_mat = jnp.concatenate([blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, 1], blocked_key_matrix[:, :, 2], blocked_key_matrix[:, :, -1], gathered_key[:, :, 0]], axis=2)
        second_value_mat = jnp.concatenate([blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, 1], blocked_value_matrix[:, :, 2], blocked_value_matrix[:, :, -1], gathered_value[:, :, 0]], axis=2)
        second_product = jnp.einsum('bhqd,bhkd->bhqk', blocked_query_matrix[:, :, 1], second_key_mat)
        second_seq_pad = jnp.concatenate([to_mask[:, :, :, :3 * to_block_size], to_mask[:, :, :, -to_block_size:], jnp.ones([bsz, 1, 1, n_rand_blocks * to_block_size], dtype=to_mask.dtype)], axis=3)
        second_rand_pad = jnp.concatenate([jnp.ones([bsz, n_heads, from_block_size, 4 * to_block_size], dtype=rand_mask.dtype), rand_mask[:, :, 0]], axis=3)
        second_product = second_product * rsqrt_d
        second_product += (1.0 - jnp.minimum(second_seq_pad, second_rand_pad)) * attn_mask_penalty
        second_attn_weights = jax.nn.softmax(second_product, axis=-1)
        second_context_layer = jnp.einsum('bhqk,bhkd->bhqd', second_attn_weights, second_value_mat)
        second_context_layer = jnp.expand_dims(second_context_layer, 2)
        exp_blocked_key_matrix = jnp.concatenate([blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2], blocked_key_matrix[:, :, 3:-1]], axis=3)
        exp_blocked_value_matrix = jnp.concatenate([blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2], blocked_value_matrix[:, :, 3:-1]], axis=3)
        middle_query_matrix = blocked_query_matrix[:, :, 2:-2]
        inner_band_product = jnp.einsum('bhlqd,bhlkd->bhlqk', middle_query_matrix, exp_blocked_key_matrix)
        inner_band_product = inner_band_product * rsqrt_d
        rand_band_product = jnp.einsum('bhlqd,bhlkd->bhlqk', middle_query_matrix, gathered_key[:, :, 1:-1])
        rand_band_product = rand_band_product * rsqrt_d
        first_band_product = jnp.einsum('bhlqd,bhkd->bhlqk', middle_query_matrix, blocked_key_matrix[:, :, 0])
        first_band_product = first_band_product * rsqrt_d
        last_band_product = jnp.einsum('bhlqd,bhkd->bhlqk', middle_query_matrix, blocked_key_matrix[:, :, -1])
        last_band_product = last_band_product * rsqrt_d
        inner_band_product += (1.0 - band_mask) * attn_mask_penalty
        first_band_product += (1.0 - jnp.expand_dims(to_mask[:, :, :, :to_block_size], 3)) * attn_mask_penalty
        last_band_product += (1.0 - jnp.expand_dims(to_mask[:, :, :, -to_block_size:], 3)) * attn_mask_penalty
        rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * attn_mask_penalty
        band_product = jnp.concatenate([first_band_product, inner_band_product, rand_band_product, last_band_product], axis=-1)
        attn_weights = jax.nn.softmax(band_product, axis=-1)
        context_layer = jnp.einsum('bhlqk,bhlkd->bhlqd', attn_weights[:, :, :, :, to_block_size:4 * to_block_size], exp_blocked_value_matrix)
        context_layer += jnp.einsum('bhlqk,bhlkd->bhlqd', attn_weights[:, :, :, :, 4 * to_block_size:-to_block_size], gathered_value[:, :, 1:-1])
        context_layer += jnp.einsum('bhlqk,bhkd->bhlqd', attn_weights[:, :, :, :, :to_block_size], blocked_value_matrix[:, :, 0])
        context_layer += jnp.einsum('bhlqk,bhkd->bhlqd', attn_weights[:, :, :, :, -to_block_size:], blocked_value_matrix[:, :, -1])
        second_last_key_mat = jnp.concatenate([blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, -3], blocked_key_matrix[:, :, -2], blocked_key_matrix[:, :, -1], gathered_key[:, :, -1]], axis=2)
        second_last_value_mat = jnp.concatenate([blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, -3], blocked_value_matrix[:, :, -2], blocked_value_matrix[:, :, -1], gathered_value[:, :, -1]], axis=2)
        second_last_product = jnp.einsum('bhqd,bhkd->bhqk', blocked_query_matrix[:, :, -2], second_last_key_mat)
        second_last_seq_pad = jnp.concatenate([to_mask[:, :, :, :to_block_size], to_mask[:, :, :, -3 * to_block_size:], jnp.ones([bsz, 1, 1, n_rand_blocks * to_block_size], dtype=to_mask.dtype)], axis=3)
        second_last_rand_pad = jnp.concatenate([jnp.ones([bsz, n_heads, from_block_size, 4 * to_block_size], dtype=rand_mask.dtype), rand_mask[:, :, -1]], axis=3)
        second_last_product = second_last_product * rsqrt_d
        second_last_product += (1.0 - jnp.minimum(second_last_seq_pad, second_last_rand_pad)) * attn_mask_penalty
        second_last_attn_weights = jax.nn.softmax(second_last_product, axis=-1)
        second_last_context_layer = jnp.einsum('bhqk,bhkd->bhqd', second_last_attn_weights, second_last_value_mat)
        second_last_context_layer = jnp.expand_dims(second_last_context_layer, 2)
        last_product = jnp.einsum('bhqd,bhkd->bhqk', blocked_query_matrix[:, :, -1], key_layer)
        last_product = last_product * rsqrt_d
        last_product += (1.0 - to_mask) * attn_mask_penalty
        last_attn_weights = jax.nn.softmax(last_product, axis=-1)
        last_context_layer = jnp.einsum('bhqk,bhkd->bhqd', last_attn_weights, value_layer)
        last_context_layer = jnp.expand_dims(last_context_layer, 2)
        context_layer = jnp.concatenate([first_context_layer, second_context_layer, context_layer, second_last_context_layer, last_context_layer], axis=2)
        context_layer = context_layer.reshape(bsz, n_heads, from_seq_len, -1) * from_mask
        context_layer = jnp.transpose(context_layer, axes=(0, 2, 1, 3)).reshape(bsz, from_seq_len, -1)
        attention_probs = None
        return (context_layer, attention_probs)

    @staticmethod
    def jax_gather(params, indices, batch_dims=2):
        """
        Gather the indices from params correctly (equivalent to tf.gather but with modifications)

        Args:
            params: (bsz, n_heads, num_blocks, block_size, head_dim)
            indices: (<num_blocks, 1)
        """

        def _jax_gather(params, indices):
            return params[indices]
        for _ in range(batch_dims):
            _jax_gather = jax.vmap(_jax_gather, in_axes=(0, 0))
        return _jax_gather(params, indices)

    def _create_rand_mask_from_inputs(self, from_blocked_mask, to_blocked_mask, broadcasted_rand_attn, num_attention_heads, num_random_blocks, batch_size, from_seq_length, from_block_size):
        """
        Create 3D attention mask from a 2D tensor mask.

        Args:
            from_blocked_mask: 2D Tensor of shape [batch_size, from_seq_length//from_block_size, from_block_size].
            to_blocked_mask: int32 Tensor of shape [batch_size, to_seq_length//to_block_size, to_block_size].
            broadcasted_rand_attn:
                [batch_size, num_attention_heads, from_seq_length//from_block_size-2, num_rand_blocks]
            num_attention_heads: int. Number of attention heads.
            num_random_blocks: int. Number of random chunks per row.
            batch_size: int. Batch size for computation.
            from_seq_length: int. length of from sequence.
            from_block_size: int. size of block in from sequence.

        Returns:
            float Tensor of shape [batch_size, num_attention_heads, from_seq_length//from_block_size-2,
            from_block_size, num_rand_blocks*to_block_size].
        """
        num_windows = from_seq_length // from_block_size - 2
        rand_mask = self.jax_gather(to_blocked_mask, broadcasted_rand_attn, batch_dims=1)
        rand_mask = rand_mask.reshape(batch_size, num_attention_heads, num_windows, num_random_blocks * from_block_size)
        rand_mask = jnp.einsum('blq,bhlk->bhlqk', from_blocked_mask[:, 1:-1], rand_mask)
        return rand_mask

    @staticmethod
    def _get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
        """
        Gives the plan of where to put random attention.

        Args:
            from_seq_length: int. length of from sequence.
            from_block_size: int. size of block in from sequence.
            num_rand_blocks: int. Number of random chunks per row.

        Returns:
            plan_from_length: ending location of from block plan_num_rand_blocks: number of random ending location for
            each block
        """
        plan_from_length = []
        plan_num_rand_blocks = []
        if 2 * num_rand_blocks + 5 < from_seq_length // from_block_size:
            plan_from_length.append(int((2 * num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(0)
        elif num_rand_blocks + 5 < from_seq_length // from_block_size:
            plan_from_length.append(int((num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks // 2)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks - num_rand_blocks // 2)
        else:
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks)
        return (plan_from_length, plan_num_rand_blocks)

    @staticmethod
    def _bigbird_block_rand_mask(from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, indices_prng_key: Optional[jax.random.PRNGKey]=None, deterministic: Optional[bool]=True, last_idx: Optional[int]=-1):
        """
        Create adjacency list of random attention.

        Args:
            from_seq_length: int. length of from sequence.
            to_seq_length: int. length of to sequence.
            from_block_size: int. size of block in from sequence.
            to_block_size: int. size of block in to sequence.
            num_rand_blocks: int. Number of random chunks per row.
            indices_prng_key: jax.random.PRNGKey. PRNG key that is used to perform random jax operations.
            deterministic: bool. When False random attention will be used.
            last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
            if positive then num_rand_blocks blocks chosen only up to last_idx.

        Returns:
            adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
        """
        if from_seq_length // from_block_size != to_seq_length // to_block_size:
            raise ValueError('Error the number of blocks needs to be same!')
        rand_attn = jnp.zeros((from_seq_length // from_block_size - 2, num_rand_blocks), dtype=jnp.int32)
        if deterministic:
            return rand_attn
        middle_seq = jnp.arange(1, to_seq_length // to_block_size - 1, dtype=jnp.int32)
        last = to_seq_length // to_block_size - 1
        if last_idx > 2 * to_block_size:
            last = last_idx // to_block_size - 1
        r = num_rand_blocks
        for i in range(1, from_seq_length // from_block_size - 1):
            start = i - 2
            end = i
            if i == 1:
                seq_values = jax.random.permutation(indices_prng_key, middle_seq[2:last])[:r]
                rand_attn = rand_attn.at[i - 1].set(seq_values)
            elif i == 2:
                seq_values = jax.random.permutation(indices_prng_key, middle_seq[3:last])[:r]
                rand_attn = rand_attn.at[i - 1].set(seq_values)
            elif i == from_seq_length // from_block_size - 3:
                seq_values = jax.random.permutation(indices_prng_key, middle_seq[:last])[:r]
                rand_attn = rand_attn.at[i - 1].set(seq_values)
            elif i == from_seq_length // from_block_size - 2:
                seq_values = jax.random.permutation(indices_prng_key, middle_seq[:last])[:r]
                rand_attn = rand_attn.at[i - 1].set(seq_values)
            elif start > last:
                start = last
                seq_values = jax.random.permutation(indices_prng_key, middle_seq[:start])[:r]
                rand_attn = rand_attn.at[i - 1].set(seq_values)
            elif end + 1 == last:
                seq_values = jax.random.permutation(indices_prng_key, middle_seq[:start])[:r]
                rand_attn = rand_attn.at[i - 1].set(seq_values)
            else:
                concat_values = jnp.concatenate((middle_seq[:start], middle_seq[end + 1:last]))
                seq_values = jax.random.permutation(indices_prng_key, concat_values)[:r]
                rand_attn = rand_attn.at[i - 1].set(seq_values)
        return rand_attn

    def _bigbird_block_rand_mask_with_head(self, from_seq_length, to_seq_length, from_block_size, to_block_size, num_heads, plan_from_length, plan_num_rand_blocks, indices_prng_key: Optional[jax.random.PRNGKey]=None, deterministic: Optional[bool]=True, window_block_left=1, window_block_right=1, global_block_top=1, global_block_bottom=1, global_block_left=1, global_block_right=1):
        """
        Create adjacency list of random attention.

        Args:
            from_seq_length: int. length of from sequence.
            to_seq_length: int. length of to sequence.
            from_block_size: int. size of block in from sequence.
            to_block_size: int. size of block in to sequence.
            num_heads: int. total number of heads.
            plan_from_length: list. plan from length where num_random_blocks are choosen from.
            plan_num_rand_blocks: list. number of rand blocks within the plan.
            indices_prng_key: jax.random.PRNGKey. PRNG key that is used to perform random jax operations.
            deterministic: bool. When False random attention will be used.
            window_block_left: int. number of blocks of window to left of a block.
            window_block_right: int. number of blocks of window to right of a block.
            global_block_top: int. number of blocks at the top.
            global_block_bottom: int. number of blocks at the bottom.
            global_block_left: int. Number of blocks globally used to the left.
            global_block_right: int. Number of blocks globally used to the right.

        Returns:
            adjacency list of size num_head where each element is of size from_seq_length//from_block_size-2 by
            num_rand_blocks
        """
        if from_seq_length // from_block_size != to_seq_length // to_block_size:
            raise ValueError('Error the number of blocks needs to be same!')
        if from_seq_length not in plan_from_length:
            raise ValueError('Error from sequence length not in plan!')
        num_blocks = from_seq_length // from_block_size
        plan_block_length = jnp.array(plan_from_length) // from_block_size
        max_plan_idx = plan_from_length.index(from_seq_length)
        rand_attn = [jnp.zeros((num_blocks, sum(plan_num_rand_blocks[:max_plan_idx + 1])), dtype=jnp.int32) for i in range(num_heads)]
        if deterministic:
            for nh in range(num_heads):
                rand_attn[nh] = rand_attn[nh][global_block_top:num_blocks - global_block_bottom, :]
            return rand_attn
        for plan_idx in range(max_plan_idx + 1):
            rnd_r_cnt = 0
            if plan_idx > 0:
                if plan_num_rand_blocks[plan_idx] > 0:
                    rnd_r_cnt = int(sum(plan_num_rand_blocks[:plan_idx]))
                    curr_r_cnt = int(sum(plan_num_rand_blocks[:plan_idx + 1]))
                    for blk_rw_idx in range(global_block_top, plan_block_length[plan_idx - 1]):
                        for h in range(num_heads):
                            single_block_row_attention = self._get_single_block_row_attention(block_id=blk_rw_idx, to_start_block_id=plan_block_length[plan_idx - 1], to_end_block_id=plan_block_length[plan_idx], num_rand_blocks=plan_num_rand_blocks[plan_idx], window_block_left=window_block_left, window_block_right=window_block_right, global_block_left=global_block_left, global_block_right=global_block_right, indices_prng_key=indices_prng_key)
                            rand_attn[h] = rand_attn[h].at[blk_rw_idx, rnd_r_cnt:curr_r_cnt].set(single_block_row_attention)
                for pl_id in range(plan_idx):
                    if plan_num_rand_blocks[pl_id] == 0:
                        continue
                    for blk_rw_idx in range(plan_block_length[plan_idx - 1], plan_block_length[plan_idx]):
                        rnd_r_cnt = 0
                        to_start_block_id = 0
                        if pl_id > 0:
                            rnd_r_cnt = int(sum(plan_num_rand_blocks[:pl_id]))
                            to_start_block_id = plan_block_length[pl_id - 1]
                        curr_r_cnt = int(sum(plan_num_rand_blocks[:pl_id + 1]))
                        for h in range(num_heads):
                            single_block_row_attention = self._get_single_block_row_attention(block_id=blk_rw_idx, to_start_block_id=to_start_block_id, to_end_block_id=plan_block_length[pl_id], num_rand_blocks=plan_num_rand_blocks[pl_id], window_block_left=window_block_left, window_block_right=window_block_right, global_block_left=global_block_left, global_block_right=global_block_right, indices_prng_key=indices_prng_key)
                            rand_attn[h] = rand_attn[h].at[blk_rw_idx, rnd_r_cnt:curr_r_cnt].set(single_block_row_attention)
            if plan_num_rand_blocks[plan_idx] == 0:
                continue
            curr_r_cnt = int(sum(plan_num_rand_blocks[:plan_idx + 1]))
            from_start_block_id = global_block_top
            to_start_block_id = 0
            if plan_idx > 0:
                rnd_r_cnt = int(sum(plan_num_rand_blocks[:plan_idx]))
                from_start_block_id = plan_block_length[plan_idx - 1]
                to_start_block_id = plan_block_length[plan_idx - 1]
            for blk_rw_idx in range(from_start_block_id, plan_block_length[plan_idx]):
                for h in range(num_heads):
                    single_block_row_attention = self._get_single_block_row_attention(block_id=blk_rw_idx, to_start_block_id=to_start_block_id, to_end_block_id=plan_block_length[plan_idx], num_rand_blocks=plan_num_rand_blocks[plan_idx], window_block_left=window_block_left, window_block_right=window_block_right, global_block_left=global_block_left, global_block_right=global_block_right, indices_prng_key=indices_prng_key)
                    rand_attn[h] = rand_attn[h].at[blk_rw_idx, rnd_r_cnt:curr_r_cnt].set(single_block_row_attention)
        for nh in range(num_heads):
            rand_attn[nh] = rand_attn[nh][global_block_top:num_blocks - global_block_bottom, :]
        return rand_attn

    @staticmethod
    def _get_single_block_row_attention(block_id, to_start_block_id, to_end_block_id, num_rand_blocks, indices_prng_key: Optional[jax.random.PRNGKey]=None, window_block_left=1, window_block_right=1, global_block_left=1, global_block_right=1):
        """
        For a single row block get random row attention.

        Args:
            block_id: int. block id of row.
            to_start_block_id: int. random attention column start id.
            to_end_block_id: int. random attention column end id.
            num_rand_blocks: int. number of random blocks to be selected.
            indices_prng_key: jax.random.PRNGKey. PRNG key that is used to perform random jax operations
            window_block_left: int. number of blocks of window to left of a block.
            window_block_right: int. number of blocks of window to right of a block.
            global_block_left: int. Number of blocks globally used to the left.
            global_block_right: int. Number of blocks globally used to the right.

        Returns:
            row containing the random attention vector of size num_rand_blocks.
        """
        to_block_list = jnp.arange(to_start_block_id, to_end_block_id, dtype=jnp.int32)
        perm_block = jax.random.permutation(indices_prng_key, to_block_list)
        illegal_blocks = list(range(block_id - window_block_left, block_id + window_block_right + 1))
        illegal_blocks.extend(list(range(global_block_left)))
        illegal_blocks.extend(list(range(to_end_block_id - global_block_right, to_end_block_id)))
        if block_id == 1:
            illegal_blocks.append(to_end_block_id - 2)
        if block_id == to_end_block_id - 2:
            illegal_blocks.append(1)
        selected_random_blocks = []
        for i in range(to_end_block_id - to_start_block_id):
            if perm_block[i] not in illegal_blocks:
                selected_random_blocks.append(perm_block[i])
            if len(selected_random_blocks) == num_rand_blocks:
                break
        return jnp.array(selected_random_blocks, dtype=jnp.int32)