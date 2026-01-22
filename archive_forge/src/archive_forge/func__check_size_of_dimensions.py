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
def _check_size_of_dimensions(self, input_shape):
    dim = input_shape[self.axis]
    if dim < self.groups:
        raise ValueError('Number of groups (' + str(self.groups) + ') cannot be more than the number of channels (' + str(dim) + ').')
    if dim % self.groups != 0:
        raise ValueError('Number of groups (' + str(self.groups) + ') must be a multiple of the number of channels (' + str(dim) + ').')