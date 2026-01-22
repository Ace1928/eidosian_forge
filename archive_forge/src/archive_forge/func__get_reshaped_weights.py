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
def _get_reshaped_weights(self, input_shape):
    broadcast_shape = self._create_broadcast_shape(input_shape)
    gamma = None
    beta = None
    if self.scale:
        gamma = tf.reshape(self.gamma, broadcast_shape)
    if self.center:
        beta = tf.reshape(self.beta, broadcast_shape)
    return (gamma, beta)