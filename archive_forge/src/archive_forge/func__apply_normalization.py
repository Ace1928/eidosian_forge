from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_wav2vec2 import Wav2Vec2Config
def _apply_normalization(self, reshaped_inputs, input_shape):
    group_shape = keras.backend.int_shape(reshaped_inputs)
    group_reduction_axes = list(range(1, len(group_shape)))
    is_instance_norm = input_shape[self.axis] // self.groups == 1
    if not is_instance_norm:
        axis = -2 if self.axis == -1 else self.axis - 1
    else:
        axis = -1 if self.axis == -1 else self.axis - 1
    group_reduction_axes.pop(axis)
    mean, variance = tf.nn.moments(reshaped_inputs, group_reduction_axes, keepdims=True)
    gamma, beta = self._get_reshaped_weights(input_shape)
    normalized_inputs = tf.nn.batch_normalization(reshaped_inputs, mean=mean, variance=variance, scale=gamma, offset=beta, variance_epsilon=self.epsilon)
    return normalized_inputs