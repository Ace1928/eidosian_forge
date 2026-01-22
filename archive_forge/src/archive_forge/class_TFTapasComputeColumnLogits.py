from __future__ import annotations
import enum
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_tapas import TapasConfig
class TFTapasComputeColumnLogits(keras.layers.Layer):

    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)
        with tf.name_scope('column_output'):
            self.column_output_weights = self.add_weight(name='column_output_weights', shape=[config.hidden_size], dtype=tf.float32, trainable=True, initializer=tf.zeros_initializer() if config.init_cell_selection_weights_to_zero else keras.initializers.TruncatedNormal(stddev=config.initializer_range))
            self.column_output_bias = self.add_weight(name='column_output_bias', shape=(), trainable=True, initializer=tf.zeros_initializer())

    def call(self, sequence_output, cell_index, cell_mask, allow_empty_column_selection) -> tf.Tensor:
        """
        Computes the column logits.

        Args:
            sequence_output (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Also known as last_hidden_state. Sequence of hidden-states at the output of the last layer of the
                model.
            cell_index (`ProductIndexMap`):
                Index that groups tokens into cells.
            cell_mask (`tf.Tensor` of shape `(batch_size, max_num_rows * max_num_cols)`):
                Mask for cells that exist in the table (i.e. that are not padding).
            allow_empty_column_selection (`bool`):
                Whether to allow not to select any column

        Returns:
            column_logits (`tf.Tensor`of shape `(batch_size, max_num_cols)`): Tensor containing the column logits for
            every example in the batch.
        """
        token_logits = tf.einsum('bsj,j->bs', sequence_output, self.column_output_weights) + self.column_output_bias
        cell_logits, cell_logits_index = reduce_mean(token_logits, cell_index)
        column_index = cell_index.project_inner(cell_logits_index)
        column_logits, out_index = reduce_sum(cell_logits * cell_mask, column_index)
        cell_count, _ = reduce_sum(cell_mask, column_index)
        column_logits /= cell_count + EPSILON_ZERO_DIVISION
        is_padding = tf.logical_and(cell_count < 0.5, tf.not_equal(out_index.indices, 0))
        column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * tf.cast(is_padding, tf.float32)
        if not allow_empty_column_selection:
            column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * tf.cast(tf.equal(out_index.indices, 0), tf.float32)
        return column_logits