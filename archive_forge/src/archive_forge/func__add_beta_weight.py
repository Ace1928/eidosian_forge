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
def _add_beta_weight(self, input_shape):
    dim = input_shape[self.axis]
    shape = (dim,)
    if self.center:
        self.beta = self.add_weight(shape=shape, name='beta', initializer=self.beta_initializer, regularizer=self.beta_regularizer, constraint=self.beta_constraint)
    else:
        self.beta = None