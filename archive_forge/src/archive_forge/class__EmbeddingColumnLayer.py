import abc
import collections
import math
import numpy as np
import six
from tensorflow.python.eager import context
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import template
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
class _EmbeddingColumnLayer(base.Layer):
    """A layer that stores all the state required for a embedding column."""

    def __init__(self, embedding_shape, initializer, weight_collections=None, trainable=True, name=None, **kwargs):
        """Constructor.

    Args:
      embedding_shape: Shape of the embedding variable used for lookup.
      initializer: A variable initializer function to be used in embedding
        variable initialization.
      weight_collections: A list of collection names to which the Variable will
        be added. Note that, variables will also be added to collections
        `tf.GraphKeys.GLOBAL_VARIABLES` and `ops.GraphKeys.MODEL_VARIABLES`.
      trainable: If `True` also add the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      name: Name of the layer
      **kwargs: keyword named properties.
    """
        super(_EmbeddingColumnLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self._embedding_shape = embedding_shape
        self._initializer = initializer
        self._weight_collections = weight_collections

    def set_weight_collections(self, weight_collections):
        """Sets the weight collections for the layer.

    Args:
      weight_collections: A list of collection names to which the Variable will
        be added.
    """
        self._weight_collections = weight_collections

    def build(self, _):
        self._embedding_weight_var = self.add_variable(name='embedding_weights', shape=self._embedding_shape, dtype=dtypes.float32, initializer=self._initializer, trainable=self.trainable)
        if self._weight_collections and (not context.executing_eagerly()):
            _add_to_collections(self._embedding_weight_var, self._weight_collections)
        self.built = True

    def call(self, _):
        return self._embedding_weight_var