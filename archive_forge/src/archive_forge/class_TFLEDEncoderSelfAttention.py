from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_led import LEDConfig
class TFLEDEncoderSelfAttention(keras.layers.Layer):

    def __init__(self, config, layer_id, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads}')
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size
        self.query = keras.layers.Dense(self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name='query')
        self.key = keras.layers.Dense(self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name='key')
        self.value = keras.layers.Dense(self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name='value')
        self.query_global = keras.layers.Dense(self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name='query_global')
        self.key_global = keras.layers.Dense(self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name='key_global')
        self.value_global = keras.layers.Dense(self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name='value_global')
        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.global_dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert attention_window % 2 == 0, f'`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}'
        assert attention_window > 0, f'`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}'
        self.one_sided_attn_window_size = attention_window // 2

    def build(self, input_shape=None):
        if not self.built:
            with tf.name_scope('query_global'):
                self.query_global.build((self.config.hidden_size,))
            with tf.name_scope('key_global'):
                self.key_global.build((self.config.hidden_size,))
            with tf.name_scope('value_global'):
                self.value_global.build((self.config.hidden_size,))
        if self.built:
            return
        self.built = True
        if getattr(self, 'query', None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        if getattr(self, 'key', None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        if getattr(self, 'value', None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
        if getattr(self, 'query_global', None) is not None:
            with tf.name_scope(self.query_global.name):
                self.query_global.build([None, None, self.config.hidden_size])
        if getattr(self, 'key_global', None) is not None:
            with tf.name_scope(self.key_global.name):
                self.key_global.build([None, None, self.config.hidden_size])
        if getattr(self, 'value_global', None) is not None:
            with tf.name_scope(self.value_global.name):
                self.value_global.build([None, None, self.config.hidden_size])

    def call(self, inputs, training=False):
        """
        LongformerSelfAttention expects *len(hidden_states)* to be multiple of *attention_window*. Padding to
        *attention_window* happens in LongformerModel.forward to avoid redoing the padding on each layer.

        The *attention_mask* is changed in [`LongformerModel.forward`] from 0, 1, 2 to:

            - -10000: no attention
            - 0: local attention
            - +10000: global attention
        """
        hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn = inputs
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)
        batch_size, seq_len, embed_dim = shape_list(hidden_states)
        tf.debugging.assert_equal(embed_dim, self.embed_dim, message=f'hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}')
        query_vectors /= tf.math.sqrt(tf.cast(self.head_dim, dtype=query_vectors.dtype))
        query_vectors = tf.reshape(query_vectors, (batch_size, seq_len, self.num_heads, self.head_dim))
        key_vectors = tf.reshape(key_vectors, (batch_size, seq_len, self.num_heads, self.head_dim))
        attn_scores = self._sliding_chunks_query_key_matmul(query_vectors, key_vectors, self.one_sided_attn_window_size)
        remove_from_windowed_attention_mask = attention_mask != 0
        float_mask = tf.cast(remove_from_windowed_attention_mask, dtype=query_vectors.dtype) * LARGE_NEGATIVE
        diagonal_mask = self._sliding_chunks_query_key_matmul(tf.ones(shape_list(attention_mask)), float_mask, self.one_sided_attn_window_size)
        attn_scores += diagonal_mask
        tf.debugging.assert_equal(shape_list(attn_scores), [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + 1], message=f'attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + 1}), but is of size {shape_list(attn_scores)}')
        max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero = self._get_global_attn_indices(is_index_global_attn)
        if is_global_attn:
            attn_scores = self._concat_with_global_key_attn_probs(attn_scores=attn_scores, query_vectors=query_vectors, key_vectors=key_vectors, max_num_global_attn_indices=max_num_global_attn_indices, is_index_global_attn_nonzero=is_index_global_attn_nonzero, is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero)
        attn_probs = stable_softmax(attn_scores, axis=-1)
        if is_global_attn:
            masked_index = tf.tile(is_index_masked[:, :, None, None], (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1))
        else:
            masked_index = tf.tile(is_index_masked[:, :, None, None], (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + 1))
        attn_probs = tf.where(masked_index, tf.zeros(shape_list(masked_index), dtype=attn_probs.dtype), attn_probs)
        if layer_head_mask is not None:
            tf.debugging.assert_equal(shape_list(layer_head_mask), [self.num_heads], message=f'Head mask for a single layer should be of size {self.num_heads}, but is {shape_list(layer_head_mask)}')
            attn_probs = tf.reshape(layer_head_mask, (1, 1, -1, 1)) * attn_probs
        attn_probs = self.dropout(attn_probs, training=training)
        value_vectors = tf.reshape(value_vectors, (batch_size, seq_len, self.num_heads, self.head_dim))
        if is_global_attn:
            attn_output = self._compute_attn_output_with_global_indices(value_vectors=value_vectors, attn_probs=attn_probs, max_num_global_attn_indices=max_num_global_attn_indices, is_index_global_attn_nonzero=is_index_global_attn_nonzero, is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero)
        else:
            attn_output = self._sliding_chunks_matmul_attn_probs_value(attn_probs, value_vectors, self.one_sided_attn_window_size)
        tf.debugging.assert_equal(shape_list(attn_output), [batch_size, seq_len, self.num_heads, self.head_dim], message='Unexpected size')
        attn_output = tf.reshape(attn_output, (batch_size, seq_len, embed_dim))
        if is_global_attn:
            attn_output, global_attn_probs = self._compute_global_attn_output_from_hidden(attn_output=attn_output, hidden_states=hidden_states, max_num_global_attn_indices=max_num_global_attn_indices, layer_head_mask=layer_head_mask, is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero, is_index_global_attn_nonzero=is_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero, is_index_masked=is_index_masked, training=training)
        else:
            global_attn_probs = tf.zeros((batch_size, self.num_heads, max_num_global_attn_indices, seq_len))
        if is_global_attn:
            masked_global_attn_index = tf.tile(is_index_global_attn[:, :, None, None], (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1))
        else:
            masked_global_attn_index = tf.tile(is_index_global_attn[:, :, None, None], (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + 1))
        attn_probs = tf.where(masked_global_attn_index, tf.zeros(shape_list(masked_global_attn_index), dtype=attn_probs.dtype), attn_probs)
        outputs = (attn_output, attn_probs, global_attn_probs)
        return outputs

    def _sliding_chunks_query_key_matmul(self, query, key, window_overlap):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This
        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer) with an
        overlap of size window_overlap
        """
        batch_size, seq_len, num_heads, head_dim = shape_list(query)
        tf.debugging.assert_equal(seq_len % (window_overlap * 2), 0, message=f'Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}')
        tf.debugging.assert_equal(shape_list(query), shape_list(key), message=f'Shape of query and key should be equal, but got query: {shape_list(query)} and key: {shape_list(key)}')
        chunks_count = seq_len // window_overlap - 1
        query = tf.reshape(tf.transpose(query, (0, 2, 1, 3)), (batch_size * num_heads, seq_len, head_dim))
        key = tf.reshape(tf.transpose(key, (0, 2, 1, 3)), (batch_size * num_heads, seq_len, head_dim))
        chunked_query = self._chunk(query, window_overlap)
        chunked_key = self._chunk(key, window_overlap)
        chunked_query = tf.cast(chunked_query, dtype=chunked_key.dtype)
        chunked_attention_scores = tf.einsum('bcxd,bcyd->bcxy', chunked_query, chunked_key)
        paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 1], [0, 0]])
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(chunked_attention_scores, paddings)
        diagonal_attn_scores_up_triang = tf.concat([diagonal_chunked_attention_scores[:, :, :window_overlap, :window_overlap + 1], diagonal_chunked_attention_scores[:, -1:, window_overlap:, :window_overlap + 1]], axis=1)
        diagonal_attn_scores_low_triang = tf.concat([tf.zeros((batch_size * num_heads, 1, window_overlap, window_overlap), dtype=diagonal_chunked_attention_scores.dtype), diagonal_chunked_attention_scores[:, :, -(window_overlap + 1):-1, window_overlap + 1:]], axis=1)
        diagonal_attn_scores_first_chunk = tf.concat([tf.roll(diagonal_chunked_attention_scores, shift=[1, window_overlap], axis=[2, 3])[:, :, :window_overlap, :window_overlap], tf.zeros((batch_size * num_heads, 1, window_overlap, window_overlap), dtype=diagonal_chunked_attention_scores.dtype)], axis=1)
        first_chunk_mask = tf.tile(tf.range(chunks_count + 1, dtype=tf.int64)[None, :, None, None], (batch_size * num_heads, 1, window_overlap, window_overlap)) < 1
        diagonal_attn_scores_low_triang = tf.where(first_chunk_mask, diagonal_attn_scores_first_chunk, diagonal_attn_scores_low_triang)
        diagonal_attention_scores = tf.concat([diagonal_attn_scores_low_triang, diagonal_attn_scores_up_triang], axis=-1)
        diagonal_attention_scores = tf.transpose(tf.reshape(diagonal_attention_scores, (batch_size, num_heads, seq_len, 2 * window_overlap + 1)), (0, 2, 1, 3))
        diagonal_attention_scores = self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    @staticmethod
    def _mask_invalid_locations(input_tensor, window_overlap):
        mask_2d_upper = tf.reverse(tf.linalg.band_part(tf.ones(shape=(window_overlap, window_overlap + 1)), -1, 0), axis=[0])
        padding = tf.convert_to_tensor([[0, shape_list(input_tensor)[1] - window_overlap], [0, shape_list(input_tensor)[3] - window_overlap - 1]])
        mask_2d = tf.pad(mask_2d_upper, padding)
        mask_2d = mask_2d + tf.reverse(mask_2d, axis=[0, 1])
        mask_4d = tf.tile(mask_2d[None, :, None, :], (shape_list(input_tensor)[0], 1, 1, 1))
        inf_tensor = -float('inf') * tf.ones_like(input_tensor)
        input_tensor = tf.where(tf.math.greater(mask_4d, 0), inf_tensor, input_tensor)
        return input_tensor

    def _sliding_chunks_matmul_attn_probs_value(self, attn_probs, value, window_overlap):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        batch_size, seq_len, num_heads, head_dim = shape_list(value)
        tf.debugging.assert_equal(seq_len % (window_overlap * 2), 0, message='Seq_len has to be multiple of 2 * window_overlap')
        tf.debugging.assert_equal(shape_list(attn_probs)[:3], shape_list(value)[:3], message='value and attn_probs must have same dims (except head_dim)')
        tf.debugging.assert_equal(shape_list(attn_probs)[3], 2 * window_overlap + 1, message='attn_probs last dim has to be 2 * window_overlap + 1')
        chunks_count = seq_len // window_overlap - 1
        chunked_attn_probs = tf.reshape(tf.transpose(attn_probs, (0, 2, 1, 3)), (batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1))
        value = tf.reshape(tf.transpose(value, (0, 2, 1, 3)), (batch_size * num_heads, seq_len, head_dim))
        paddings = tf.convert_to_tensor([[0, 0], [window_overlap, window_overlap], [0, 0]])
        padded_value = tf.pad(value, paddings, constant_values=-1)
        frame_size = 3 * window_overlap * head_dim
        frame_hop_size = (shape_list(padded_value)[1] * head_dim - frame_size) // chunks_count
        chunked_value = tf.signal.frame(tf.reshape(padded_value, (batch_size * num_heads, -1)), frame_size, frame_hop_size)
        chunked_value = tf.reshape(chunked_value, (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim))
        tf.debugging.assert_equal(shape_list(chunked_value), [batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim], message='Chunked value has the wrong shape')
        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)
        context = tf.einsum('bcwd,bcdh->bcwh', chunked_attn_probs, chunked_value)
        context = tf.transpose(tf.reshape(context, (batch_size, num_heads, seq_len, head_dim)), (0, 2, 1, 3))
        return context

    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, paddings):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = tf.pad(hidden_states_padded, paddings)
        batch_size, chunk_size, seq_length, hidden_dim = shape_list(hidden_states_padded)
        hidden_states_padded = tf.reshape(hidden_states_padded, (batch_size, chunk_size, hidden_dim, seq_length))
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
        total_num_heads, num_chunks, window_overlap, hidden_dim = shape_list(chunked_hidden_states)
        paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [0, window_overlap + 1]])
        chunked_hidden_states = tf.pad(chunked_hidden_states, paddings)
        chunked_hidden_states = tf.reshape(chunked_hidden_states, (total_num_heads, num_chunks, -1))
        chunked_hidden_states = chunked_hidden_states[:, :, :-window_overlap]
        chunked_hidden_states = tf.reshape(chunked_hidden_states, (total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim))
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        batch_size, seq_length, hidden_dim = shape_list(hidden_states)
        num_output_chunks = 2 * (seq_length // (2 * window_overlap)) - 1
        frame_hop_size = window_overlap * hidden_dim
        frame_size = 2 * frame_hop_size
        hidden_states = tf.reshape(hidden_states, (batch_size, seq_length * hidden_dim))
        chunked_hidden_states = tf.signal.frame(hidden_states, frame_size, frame_hop_size)
        tf.debugging.assert_equal(shape_list(chunked_hidden_states), [batch_size, num_output_chunks, frame_size], message=f'Make sure chunking is correctly applied. `Chunked hidden states should have output  dimension {[batch_size, frame_size, num_output_chunks]}, but got {shape_list(chunked_hidden_states)}.')
        chunked_hidden_states = tf.reshape(chunked_hidden_states, (batch_size, num_output_chunks, 2 * window_overlap, hidden_dim))
        return chunked_hidden_states

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """compute global attn indices required throughout forward pass"""
        num_global_attn_indices = tf.math.count_nonzero(is_index_global_attn, axis=1)
        num_global_attn_indices = tf.cast(num_global_attn_indices, dtype=tf.constant(1).dtype)
        max_num_global_attn_indices = tf.reduce_max(num_global_attn_indices)
        is_index_global_attn_nonzero = tf.where(is_index_global_attn)
        is_local_index_global_attn = tf.range(max_num_global_attn_indices) < tf.expand_dims(num_global_attn_indices, axis=-1)
        is_local_index_global_attn_nonzero = tf.where(is_local_index_global_attn)
        is_local_index_no_global_attn_nonzero = tf.where(tf.math.logical_not(is_local_index_global_attn))
        return (max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero)

    def _concat_with_global_key_attn_probs(self, attn_scores, key_vectors, query_vectors, max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero):
        batch_size = shape_list(key_vectors)[0]
        global_key_vectors = tf.gather_nd(key_vectors, is_index_global_attn_nonzero)
        key_vectors_only_global = tf.scatter_nd(is_local_index_global_attn_nonzero, global_key_vectors, shape=(batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim))
        attn_probs_from_global_key = tf.einsum('blhd,bshd->blhs', query_vectors, key_vectors_only_global)
        attn_probs_from_global_key_trans = tf.transpose(attn_probs_from_global_key, (0, 3, 1, 2))
        mask_shape = (shape_list(is_local_index_no_global_attn_nonzero)[0],) + tuple(shape_list(attn_probs_from_global_key_trans)[-2:])
        mask = tf.ones(mask_shape) * -10000.0
        mask = tf.cast(mask, dtype=attn_probs_from_global_key_trans.dtype)
        attn_probs_from_global_key_trans = tf.tensor_scatter_nd_update(attn_probs_from_global_key_trans, is_local_index_no_global_attn_nonzero, mask)
        attn_probs_from_global_key = tf.transpose(attn_probs_from_global_key_trans, (0, 2, 3, 1))
        attn_scores = tf.concat((attn_probs_from_global_key, attn_scores), axis=-1)
        return attn_scores

    def _compute_attn_output_with_global_indices(self, value_vectors, attn_probs, max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero):
        batch_size = shape_list(attn_probs)[0]
        attn_probs_only_global = attn_probs[:, :, :, :max_num_global_attn_indices]
        global_value_vectors = tf.gather_nd(value_vectors, is_index_global_attn_nonzero)
        value_vectors_only_global = tf.scatter_nd(is_local_index_global_attn_nonzero, global_value_vectors, shape=(batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim))
        attn_output_only_global = tf.einsum('blhs,bshd->blhd', attn_probs_only_global, value_vectors_only_global)
        attn_probs_without_global = attn_probs[:, :, :, max_num_global_attn_indices:]
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(attn_probs_without_global, value_vectors, self.one_sided_attn_window_size)
        return attn_output_only_global + attn_output_without_global

    def _compute_global_attn_output_from_hidden(self, attn_output, hidden_states, max_num_global_attn_indices, layer_head_mask, is_local_index_global_attn_nonzero, is_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero, is_index_masked, training):
        batch_size, seq_len = shape_list(hidden_states)[:2]
        global_attn_hidden_states = tf.gather_nd(hidden_states, is_index_global_attn_nonzero)
        global_attn_hidden_states = tf.scatter_nd(is_local_index_global_attn_nonzero, global_attn_hidden_states, shape=(batch_size, max_num_global_attn_indices, self.embed_dim))
        global_query_vectors_only_global = self.query_global(global_attn_hidden_states)
        global_key_vectors = self.key_global(hidden_states)
        global_value_vectors = self.value_global(hidden_states)
        global_query_vectors_only_global /= tf.math.sqrt(tf.cast(self.head_dim, dtype=global_query_vectors_only_global.dtype))
        global_query_vectors_only_global = self.reshape_and_transpose(global_query_vectors_only_global, batch_size)
        global_key_vectors = self.reshape_and_transpose(global_key_vectors, batch_size)
        global_value_vectors = self.reshape_and_transpose(global_value_vectors, batch_size)
        global_attn_scores = tf.matmul(global_query_vectors_only_global, global_key_vectors, transpose_b=True)
        tf.debugging.assert_equal(shape_list(global_attn_scores), [batch_size * self.num_heads, max_num_global_attn_indices, seq_len], message=f'global_attn_scores have the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)}, but is {shape_list(global_attn_scores)}.')
        global_attn_scores = tf.reshape(global_attn_scores, (batch_size, self.num_heads, max_num_global_attn_indices, seq_len))
        global_attn_scores_trans = tf.transpose(global_attn_scores, (0, 2, 1, 3))
        mask_shape = (shape_list(is_local_index_no_global_attn_nonzero)[0],) + tuple(shape_list(global_attn_scores_trans)[-2:])
        global_attn_mask = tf.ones(mask_shape) * -10000.0
        global_attn_mask = tf.cast(global_attn_mask, dtype=global_attn_scores_trans.dtype)
        global_attn_scores_trans = tf.tensor_scatter_nd_update(global_attn_scores_trans, is_local_index_no_global_attn_nonzero, global_attn_mask)
        global_attn_scores = tf.transpose(global_attn_scores_trans, (0, 2, 1, 3))
        attn_mask = tf.tile(is_index_masked[:, None, None, :], (1, shape_list(global_attn_scores)[1], 1, 1))
        global_attn_scores = tf.where(attn_mask, -10000.0, global_attn_scores)
        global_attn_scores = tf.reshape(global_attn_scores, (batch_size * self.num_heads, max_num_global_attn_indices, seq_len))
        global_attn_probs_float = stable_softmax(global_attn_scores, axis=-1)
        if layer_head_mask is not None:
            tf.debugging.assert_equal(shape_list(layer_head_mask), [self.num_heads], message=f'Head mask for a single layer should be of size {self.num_heads}, but is {shape_list(layer_head_mask)}')
            global_attn_probs_float = tf.reshape(layer_head_mask, (1, -1, 1, 1)) * tf.reshape(global_attn_probs_float, (batch_size, self.num_heads, max_num_global_attn_indices, seq_len))
            global_attn_probs_float = tf.reshape(global_attn_probs_float, (batch_size * self.num_heads, max_num_global_attn_indices, seq_len))
        global_attn_probs = self.global_dropout(global_attn_probs_float, training=training)
        global_attn_output = tf.matmul(global_attn_probs, global_value_vectors)
        tf.debugging.assert_equal(shape_list(global_attn_output), [batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim], message=f'global_attn_output tensor has the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim)}, but is {shape_list(global_attn_output)}.')
        global_attn_output = tf.reshape(global_attn_output, (batch_size, self.num_heads, max_num_global_attn_indices, self.head_dim))
        nonzero_global_attn_output = tf.gather_nd(tf.transpose(global_attn_output, (0, 2, 1, 3)), is_local_index_global_attn_nonzero)
        nonzero_global_attn_output = tf.reshape(nonzero_global_attn_output, (shape_list(is_local_index_global_attn_nonzero)[0], -1))
        attn_output = tf.tensor_scatter_nd_update(attn_output, is_index_global_attn_nonzero, nonzero_global_attn_output)
        global_attn_probs = tf.reshape(global_attn_probs, (batch_size, self.num_heads, max_num_global_attn_indices, seq_len))
        return (attn_output, global_attn_probs)

    def reshape_and_transpose(self, vector, batch_size):
        return tf.reshape(tf.transpose(tf.reshape(vector, (batch_size, -1, self.num_heads, self.head_dim)), (0, 2, 1, 3)), (batch_size * self.num_heads, -1, self.head_dim))