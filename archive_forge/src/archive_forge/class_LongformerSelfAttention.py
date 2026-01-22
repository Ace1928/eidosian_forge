import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_longformer import LongformerConfig
class LongformerSelfAttention(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)
        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.dropout = config.attention_probs_dropout_prob
        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert attention_window % 2 == 0, f'`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}'
        assert attention_window > 0, f'`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}'
        self.one_sided_attn_window_size = attention_window // 2
        self.config = config

    def forward(self, hidden_states, attention_mask=None, layer_head_mask=None, is_index_masked=None, is_index_global_attn=None, is_global_attn=None, output_attentions=False):
        """
        [`LongformerSelfAttention`] expects *len(hidden_states)* to be multiple of *attention_window*. Padding to
        *attention_window* happens in [`LongformerModel.forward`] to avoid redoing the padding on each layer.

        The *attention_mask* is changed in [`LongformerModel.forward`] from 0, 1, 2 to:

            - -10000: no attention
            - 0: local attention
            - +10000: global attention
        """
        hidden_states = hidden_states.transpose(0, 1)
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)
        seq_len, batch_size, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim, f'hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}'
        query_vectors /= math.sqrt(self.head_dim)
        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        attn_scores = self._sliding_chunks_query_key_matmul(query_vectors, key_vectors, self.one_sided_attn_window_size)
        remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(remove_from_windowed_attention_mask, torch.finfo(query_vectors.dtype).min)
        diagonal_mask = self._sliding_chunks_query_key_matmul(float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size)
        attn_scores += diagonal_mask
        assert list(attn_scores.size()) == [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + 1], f'local_attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}'
        if is_global_attn:
            max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero = self._get_global_attn_indices(is_index_global_attn)
            global_key_attn_scores = self._concat_with_global_key_attn_probs(query_vectors=query_vectors, key_vectors=key_vectors, max_num_global_attn_indices=max_num_global_attn_indices, is_index_global_attn_nonzero=is_index_global_attn_nonzero, is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero)
            attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)
            del global_key_attn_scores
        attn_probs = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32)
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (self.num_heads,), f'Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}'
            attn_probs = layer_head_mask.view(1, 1, -1, 1) * attn_probs
        attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        attn_probs = attn_probs.type_as(attn_scores)
        del attn_scores
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        if is_global_attn:
            attn_output = self._compute_attn_output_with_global_indices(value_vectors=value_vectors, attn_probs=attn_probs, max_num_global_attn_indices=max_num_global_attn_indices, is_index_global_attn_nonzero=is_index_global_attn_nonzero, is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero)
        else:
            attn_output = self._sliding_chunks_matmul_attn_probs_value(attn_probs, value_vectors, self.one_sided_attn_window_size)
        assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), 'Unexpected size'
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        if is_global_attn:
            global_attn_output, global_attn_probs = self._compute_global_attn_output_from_hidden(hidden_states=hidden_states, max_num_global_attn_indices=max_num_global_attn_indices, layer_head_mask=layer_head_mask, is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero, is_index_global_attn_nonzero=is_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero, is_index_masked=is_index_masked)
            nonzero_global_attn_output = global_attn_output[is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]]
            attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.view(len(is_local_index_global_attn_nonzero[0]), -1)
            attn_probs[is_index_global_attn_nonzero] = 0
        outputs = (attn_output.transpose(0, 1),)
        if output_attentions:
            outputs += (attn_probs,)
        return outputs + (global_attn_probs,) if is_global_attn and output_attentions else outputs

    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = nn.functional.pad(hidden_states_padded, padding)
        hidden_states_padded = hidden_states_padded.view(*hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2))
        return hidden_states_padded

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        shift every row 1 step right, converting columns into diagonals.

        Example:

        ```python
        chunked_hidden_states: [
            0.4983,
            2.6918,
            -0.0071,
            1.0492,
            -1.8348,
            0.7672,
            0.2986,
            0.0285,
            -0.7584,
            0.4206,
            -0.0405,
            0.1599,
            2.0514,
            -1.1600,
            0.5372,
            0.2629,
        ]
        window_overlap = num_rows = 4
        ```

                     (pad & diagonalize) => [ 0.4983, 2.6918, -0.0071, 1.0492, 0.0000, 0.0000, 0.0000
                       0.0000, -1.8348, 0.7672, 0.2986, 0.0285, 0.0000, 0.0000 0.0000, 0.0000, -0.7584, 0.4206,
                       -0.0405, 0.1599, 0.0000 0.0000, 0.0000, 0.0000, 2.0514, -1.1600, 0.5372, 0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.size()
        chunked_hidden_states = nn.functional.pad(chunked_hidden_states, (0, window_overlap + 1))
        chunked_hidden_states = chunked_hidden_states.view(total_num_heads, num_chunks, -1)
        chunked_hidden_states = chunked_hidden_states[:, :, :-window_overlap]
        chunked_hidden_states = chunked_hidden_states.view(total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim)
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap, onnx_export: bool=False):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        if not onnx_export:
            hidden_states = hidden_states.view(hidden_states.size(0), torch.div(hidden_states.size(1), window_overlap * 2, rounding_mode='trunc'), window_overlap * 2, hidden_states.size(2))
            chunk_size = list(hidden_states.size())
            chunk_size[1] = chunk_size[1] * 2 - 1
            chunk_stride = list(hidden_states.stride())
            chunk_stride[1] = chunk_stride[1] // 2
            return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        chunk_size = [hidden_states.size(0), torch.div(hidden_states.size(1), window_overlap, rounding_mode='trunc') - 1, window_overlap * 2, hidden_states.size(2)]
        overlapping_chunks = torch.empty(chunk_size, device=hidden_states.device)
        for chunk in range(chunk_size[1]):
            overlapping_chunks[:, chunk, :, :] = hidden_states[:, chunk * window_overlap:chunk * window_overlap + 2 * window_overlap, :]
        return overlapping_chunks

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len) -> torch.Tensor:
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, :affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        input_tensor[:, :affected_seq_len, :, :affected_seq_len + 1] = torch.full_like(beginning_input, -float('inf')).where(beginning_mask.bool(), beginning_input)
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = ending_mask.expand(ending_input.size())
        input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):] = torch.full_like(ending_input, -float('inf')).where(ending_mask.bool(), ending_input)

    def _sliding_chunks_query_key_matmul(self, query: torch.Tensor, key: torch.Tensor, window_overlap: int):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This
        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer) with an
        overlap of size window_overlap
        """
        batch_size, seq_len, num_heads, head_dim = query.size()
        assert seq_len % (window_overlap * 2) == 0, f'Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}'
        assert query.size() == key.size()
        chunks_count = torch.div(seq_len, window_overlap, rounding_mode='trunc') - 1
        query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        query = self._chunk(query, window_overlap, getattr(self.config, 'onnx_export', False))
        key = self._chunk(key, window_overlap, getattr(self.config, 'onnx_export', False))
        diagonal_chunked_attention_scores = torch.einsum('bcxd,bcyd->bcxy', (query, key))
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(diagonal_chunked_attention_scores, padding=(0, 0, 0, 1))
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros((batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1))
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[:, :, :window_overlap, :window_overlap + 1]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[:, -1, window_overlap:, :window_overlap + 1]
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[:, :, -(window_overlap + 1):-1, window_overlap + 1:]
        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[:, 0, :window_overlap - 1, 1 - window_overlap:]
        diagonal_attention_scores = diagonal_attention_scores.view(batch_size, num_heads, seq_len, 2 * window_overlap + 1).transpose(2, 1)
        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(self, attn_probs: torch.Tensor, value: torch.Tensor, window_overlap: int):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        batch_size, seq_len, num_heads, head_dim = value.size()
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size()[:3] == value.size()[:3]
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = torch.div(seq_len, window_overlap, rounding_mode='trunc') - 1
        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(batch_size * num_heads, torch.div(seq_len, window_overlap, rounding_mode='trunc'), window_overlap, 2 * window_overlap + 1)
        value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (chunked_value_stride[0], window_overlap * chunked_value_stride[1], chunked_value_stride[1], chunked_value_stride[2])
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)
        context = torch.einsum('bcwd,bcdh->bcwh', (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """compute global attn indices required throughout forward pass"""
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)
        max_num_global_attn_indices = num_global_attn_indices.max()
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)
        is_local_index_global_attn = torch.arange(max_num_global_attn_indices, device=is_index_global_attn.device) < num_global_attn_indices.unsqueeze(dim=-1)
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)
        is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(as_tuple=True)
        return (max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero)

    def _concat_with_global_key_attn_probs(self, key_vectors, query_vectors, max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero):
        batch_size = key_vectors.shape[0]
        key_vectors_only_global = key_vectors.new_zeros(batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim)
        key_vectors_only_global[is_local_index_global_attn_nonzero] = key_vectors[is_index_global_attn_nonzero]
        attn_probs_from_global_key = torch.einsum('blhd,bshd->blhs', (query_vectors, key_vectors_only_global))
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)
        attn_probs_from_global_key[is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :] = torch.finfo(attn_probs_from_global_key.dtype).min
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)
        return attn_probs_from_global_key

    def _compute_attn_output_with_global_indices(self, value_vectors, attn_probs, max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero):
        batch_size = attn_probs.shape[0]
        attn_probs_only_global = attn_probs.narrow(-1, 0, max_num_global_attn_indices)
        value_vectors_only_global = value_vectors.new_zeros(batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim)
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value_vectors[is_index_global_attn_nonzero]
        attn_output_only_global = torch.matmul(attn_probs_only_global.transpose(1, 2).clone(), value_vectors_only_global.transpose(1, 2).clone()).transpose(1, 2)
        attn_probs_without_global = attn_probs.narrow(-1, max_num_global_attn_indices, attn_probs.size(-1) - max_num_global_attn_indices).contiguous()
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(attn_probs_without_global, value_vectors, self.one_sided_attn_window_size)
        return attn_output_only_global + attn_output_without_global

    def _compute_global_attn_output_from_hidden(self, hidden_states, max_num_global_attn_indices, layer_head_mask, is_local_index_global_attn_nonzero, is_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero, is_index_masked):
        seq_len, batch_size = hidden_states.shape[:2]
        global_attn_hidden_states = hidden_states.new_zeros(max_num_global_attn_indices, batch_size, self.embed_dim)
        global_attn_hidden_states[is_local_index_global_attn_nonzero[::-1]] = hidden_states[is_index_global_attn_nonzero[::-1]]
        global_query_vectors_only_global = self.query_global(global_attn_hidden_states)
        global_key_vectors = self.key_global(hidden_states)
        global_value_vectors = self.value_global(hidden_states)
        global_query_vectors_only_global /= math.sqrt(self.head_dim)
        global_query_vectors_only_global = global_query_vectors_only_global.contiguous().view(max_num_global_attn_indices, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        global_key_vectors = global_key_vectors.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        global_value_vectors = global_value_vectors.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        global_attn_scores = torch.bmm(global_query_vectors_only_global, global_key_vectors.transpose(1, 2))
        assert list(global_attn_scores.size()) == [batch_size * self.num_heads, max_num_global_attn_indices, seq_len], f'global_attn_scores have the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)}, but is {global_attn_scores.size()}.'
        global_attn_scores = global_attn_scores.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
        global_attn_scores = global_attn_scores.transpose(1, 2)
        global_attn_scores[is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :] = torch.finfo(global_attn_scores.dtype).min
        global_attn_scores = global_attn_scores.transpose(1, 2)
        global_attn_scores = global_attn_scores.masked_fill(is_index_masked[:, None, None, :], torch.finfo(global_attn_scores.dtype).min)
        global_attn_scores = global_attn_scores.view(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)
        global_attn_probs_float = nn.functional.softmax(global_attn_scores, dim=-1, dtype=torch.float32)
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (self.num_heads,), f'Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}'
            global_attn_probs_float = layer_head_mask.view(1, -1, 1, 1) * global_attn_probs_float.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
            global_attn_probs_float = global_attn_probs_float.view(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)
        global_attn_probs = nn.functional.dropout(global_attn_probs_float.type_as(global_attn_scores), p=self.dropout, training=self.training)
        global_attn_output = torch.bmm(global_attn_probs, global_value_vectors)
        assert list(global_attn_output.size()) == [batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim], f'global_attn_output tensor has the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim)}, but is {global_attn_output.size()}.'
        global_attn_probs = global_attn_probs.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
        global_attn_output = global_attn_output.view(batch_size, self.num_heads, max_num_global_attn_indices, self.head_dim)
        return (global_attn_output, global_attn_probs)