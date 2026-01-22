import collections
import math
import string
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.engine.base_layer import Layer
from keras.src.layers import activation
from keras.src.layers import core
from keras.src.layers import regularization
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def _build_from_signature(self, query, value, key=None):
    """Builds layers and variables.

        Once the method is called, self._built_from_signature will be set to
        True.

        Args:
            query: Query tensor or TensorShape.
            value: Value tensor or TensorShape.
            key: Key tensor or TensorShape.
        """
    self._built_from_signature = True
    if hasattr(query, 'shape'):
        self._query_shape = tf.TensorShape(query.shape)
    else:
        self._query_shape = tf.TensorShape(query)
    if hasattr(value, 'shape'):
        self._value_shape = tf.TensorShape(value.shape)
    else:
        self._value_shape = tf.TensorShape(value)
    if key is None:
        self._key_shape = self._value_shape
    elif hasattr(key, 'shape'):
        self._key_shape = tf.TensorShape(key.shape)
    else:
        self._key_shape = tf.TensorShape(key)
    with tf_utils.maybe_init_scope(self):
        free_dims = self._query_shape.rank - 1
        einsum_equation, bias_axes, output_rank = _build_proj_equation(free_dims, bound_dims=1, output_dims=2)
        self._query_dense = core.EinsumDense(einsum_equation, output_shape=_get_output_shape(output_rank - 1, [self._num_heads, self._key_dim]), bias_axes=bias_axes if self._use_bias else None, name='query', **self._get_common_kwargs_for_sublayer())
        einsum_equation, bias_axes, output_rank = _build_proj_equation(self._key_shape.rank - 1, bound_dims=1, output_dims=2)
        self._key_dense = core.EinsumDense(einsum_equation, output_shape=_get_output_shape(output_rank - 1, [self._num_heads, self._key_dim]), bias_axes=bias_axes if self._use_bias else None, name='key', **self._get_common_kwargs_for_sublayer())
        einsum_equation, bias_axes, output_rank = _build_proj_equation(self._value_shape.rank - 1, bound_dims=1, output_dims=2)
        self._value_dense = core.EinsumDense(einsum_equation, output_shape=_get_output_shape(output_rank - 1, [self._num_heads, self._value_dim]), bias_axes=bias_axes if self._use_bias else None, name='value', **self._get_common_kwargs_for_sublayer())
        self._build_attention(output_rank)
        self._output_dense = self._make_output_dense(free_dims, self._get_common_kwargs_for_sublayer(), 'attention_output')