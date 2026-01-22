from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxMaskedLMOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, logging
from .configuration_opt import OPTConfig
class FlaxOPTDecoder(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float32
    offset: int = 2

    def setup(self):
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        embed_dim = self.config.hidden_size
        self.padding_idx = self.config.pad_token_id
        self.max_target_positions = self.config.max_position_embeddings
        self.embed_tokens = nn.Embed(self.config.vocab_size, self.config.word_embed_proj_dim, embedding_init=jax.nn.initializers.normal(self.config.init_std), dtype=self.dtype)
        self.embed_positions = FlaxOPTLearnedPositionalEmbedding(self.config.max_position_embeddings, embed_dim, embedding_init=jax.nn.initializers.normal(self.config.init_std), dtype=self.dtype)
        if self.config.word_embed_proj_dim != self.config.hidden_size:
            self.project_in = nn.Dense(self.config.hidden_size, use_bias=False)
            self.project_out = nn.Dense(self.config.word_embed_proj_dim, use_bias=False)
        else:
            self.project_in = None
            self.project_out = None
        if self.config.do_layer_norm_before and (not self.config._remove_final_layer_norm):
            self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        else:
            self.final_layer_norm = None
        self.layers = FlaxOPTDecoderLayerCollection(self.config, self.dtype)

    def __call__(self, input_ids, attention_mask, position_ids, init_cache: bool=False, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True, deterministic: bool=True):
        input_shape = input_ids.shape
        input_ids = input_ids.reshape(-1, input_shape[-1])
        inputs_embeds = self.embed_tokens(input_ids)
        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)
        positions = self.embed_positions(position_ids)
        hidden_states = inputs_embeds + positions
        hidden_state, all_hidden_states, attentions = self.layers(hidden_states, attention_mask, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        if self.final_layer_norm is not None:
            hidden_state = self.final_layer_norm(hidden_state)
        if self.project_out is not None:
            hidden_state = self.project_out(hidden_state)
        if output_hidden_states:
            all_hidden_states += (hidden_state,)
        outputs = [hidden_state, all_hidden_states, attentions]
        if not return_dict:
            return tuple((v for v in outputs if v is not None))
        return FlaxBaseModelOutput(last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=attentions)