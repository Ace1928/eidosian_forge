from __future__ import annotations
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
from .configuration_albert import AlbertConfig
class TFAlbertPreTrainingLoss:
    """
    Loss function suitable for ALBERT pretraining, that is, the task of pretraining a language model by combining SOP +
    MLM. .. note:: Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.
    """

    def hf_compute_loss(self, labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        if self.config.tf_legacy_loss:
            masked_lm_active_loss = tf.not_equal(tf.reshape(tensor=labels['labels'], shape=(-1,)), -100)
            masked_lm_reduced_logits = tf.boolean_mask(tensor=tf.reshape(tensor=logits[0], shape=(-1, shape_list(logits[0])[2])), mask=masked_lm_active_loss)
            masked_lm_labels = tf.boolean_mask(tensor=tf.reshape(tensor=labels['labels'], shape=(-1,)), mask=masked_lm_active_loss)
            sentence_order_active_loss = tf.not_equal(tf.reshape(tensor=labels['sentence_order_label'], shape=(-1,)), -100)
            sentence_order_reduced_logits = tf.boolean_mask(tensor=tf.reshape(tensor=logits[1], shape=(-1, 2)), mask=sentence_order_active_loss)
            sentence_order_label = tf.boolean_mask(tensor=tf.reshape(tensor=labels['sentence_order_label'], shape=(-1,)), mask=sentence_order_active_loss)
            masked_lm_loss = loss_fn(y_true=masked_lm_labels, y_pred=masked_lm_reduced_logits)
            sentence_order_loss = loss_fn(y_true=sentence_order_label, y_pred=sentence_order_reduced_logits)
            masked_lm_loss = tf.reshape(tensor=masked_lm_loss, shape=(-1, shape_list(sentence_order_loss)[0]))
            masked_lm_loss = tf.reduce_mean(input_tensor=masked_lm_loss, axis=0)
            return masked_lm_loss + sentence_order_loss
        unmasked_lm_losses = loss_fn(y_true=tf.nn.relu(labels['labels']), y_pred=logits[0])
        lm_loss_mask = tf.cast(labels['labels'] != -100, dtype=unmasked_lm_losses.dtype)
        masked_lm_losses = unmasked_lm_losses * lm_loss_mask
        reduced_masked_lm_loss = tf.reduce_sum(masked_lm_losses) / tf.reduce_sum(lm_loss_mask)
        sop_logits = tf.reshape(logits[1], (-1, 2))
        unmasked_sop_loss = loss_fn(y_true=tf.nn.relu(labels['sentence_order_label']), y_pred=sop_logits)
        sop_loss_mask = tf.cast(labels['sentence_order_label'] != -100, dtype=unmasked_sop_loss.dtype)
        masked_sop_loss = unmasked_sop_loss * sop_loss_mask
        reduced_masked_sop_loss = tf.reduce_sum(masked_sop_loss) / tf.reduce_sum(sop_loss_mask)
        return tf.reshape(reduced_masked_lm_loss + reduced_masked_sop_loss, (1,))