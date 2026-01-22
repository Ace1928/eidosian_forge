import collections
import math
import os
import re
import unicodedata
from typing import List
import numpy as np
import six
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from autokeras import constants
from autokeras.utils import data_utils
@keras.utils.register_keras_serializable()
class DenseEinsum(layers.Layer):
    """from official.nlp.modeling.layers.dense_einsum.DenseEinsum"""

    def __init__(self, output_shape, num_summed_dimensions=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
        super(DenseEinsum, self).__init__(**kwargs)
        self._output_shape = output_shape if isinstance(output_shape, (list, tuple)) else (output_shape,)
        self._activation = keras.activations.get(activation)
        self._use_bias = use_bias
        self._kernel_initializer = keras.initializers.get(kernel_initializer)
        self._bias_initializer = keras.initializers.get(bias_initializer)
        self._kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = keras.constraints.get(kernel_constraint)
        self._bias_constraint = keras.constraints.get(bias_constraint)
        self._num_summed_dimensions = num_summed_dimensions
        self._einsum_string = None

    def _build_einsum_string(self, free_input_dims, bound_dims, output_dims):
        _CHR_IDX = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']
        input_str = ''
        kernel_str = ''
        output_str = ''
        letter_offset = 0
        for i in range(free_input_dims):
            char = _CHR_IDX[i + letter_offset]
            input_str += char
            output_str += char
        letter_offset += free_input_dims
        for i in range(bound_dims):
            char = _CHR_IDX[i + letter_offset]
            input_str += char
            kernel_str += char
        letter_offset += bound_dims
        for i in range(output_dims):
            char = _CHR_IDX[i + letter_offset]
            kernel_str += char
            output_str += char
        return input_str + ',' + kernel_str + '->' + output_str

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_rank = input_shape.rank
        free_input_dims = input_rank - self._num_summed_dimensions
        output_dims = len(self._output_shape)
        self._einsum_string = self._build_einsum_string(free_input_dims, self._num_summed_dimensions, output_dims)
        self._kernel_shape = input_shape[free_input_dims:].concatenate(self._output_shape)
        self._kernel = self.add_weight('kernel', shape=self._kernel_shape, initializer=self._kernel_initializer, regularizer=self._kernel_regularizer, constraint=self._kernel_constraint, dtype=self.dtype, trainable=True)
        if self._use_bias:
            self._bias = self.add_weight('bias', shape=self._output_shape, initializer=self._bias_initializer, regularizer=self._bias_regularizer, constraint=self._bias_constraint, dtype=self.dtype, trainable=True)
        else:
            self._bias = None
        super(DenseEinsum, self).build(input_shape)

    def get_config(self):
        config = {'output_shape': self._output_shape, 'num_summed_dimensions': self._num_summed_dimensions, 'activation': keras.activations.serialize(self._activation), 'use_bias': self._use_bias, 'kernel_initializer': keras.initializers.serialize(self._kernel_initializer), 'bias_initializer': keras.initializers.serialize(self._bias_initializer), 'kernel_regularizer': keras.regularizers.serialize(self._kernel_regularizer), 'bias_regularizer': keras.regularizers.serialize(self._bias_regularizer), 'activity_regularizer': keras.regularizers.serialize(self._activity_regularizer), 'kernel_constraint': keras.constraints.serialize(self._kernel_constraint), 'bias_constraint': keras.constraints.serialize(self._bias_constraint)}
        base_config = super(DenseEinsum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        ret = tf.einsum(self._einsum_string, inputs, self._kernel)
        if self._use_bias:
            ret += self._bias
        if self._activation is not None:
            ret = self._activation(ret)
        return ret