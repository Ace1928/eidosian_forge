from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ....modeling_tf_utils import (
from ....tf_utils import shape_list, stable_softmax
from ....utils import (
from .configuration_transfo_xl import TransfoXLConfig
from .modeling_tf_transfo_xl_utilities import TFAdaptiveSoftmaxMask
class TFTransfoEmbeddings(keras.layers.Layer):

    def __init__(self, vocab_size, emb_size, init_std, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.init_std = init_std

    def build(self, input_shape):
        self.weight = self.add_weight(shape=(self.vocab_size, self.emb_size), initializer=get_initializer(self.init_std), name='embeddings')
        super().build(input_shape)

    def call(self, inputs):
        return tf.gather(self.weight, inputs)