import collections
import itertools
import json
import os
import sys
import threading
import warnings
import weakref
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session as session_module
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.eager.context import get_config
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients as gradients_module
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import map_fn as map_fn_lib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import moving_averages
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
class ContextValueCache(weakref.WeakKeyDictionary):
    """Container that caches (possibly tensor) values based on the context.

  This class is similar to defaultdict, where values may be produced by the
  default factory specified during initialization. This class also has a default
  value for the key (when key is `None`) -- the key is set to the current graph
  or eager context. The default factories for key and value are only used in
  `__getitem__` and `setdefault`. The `.get()` behavior remains the same.

  This object will return the value of the current graph or closest parent graph
  if the current graph is a function. This is to reflect the fact that if a
  tensor is created in eager/graph, child functions may capture that tensor.

  The default factory method may accept keyword arguments (unlike defaultdict,
  which only accepts callables with 0 arguments). To pass keyword arguments to
  `default_factory`, use the `setdefault` method instead of `__getitem__`.

  An example of how this class can be used in different contexts:

  ```
  cache = ContextValueCache(int)

  # Eager mode
  cache[None] += 2
  cache[None] += 4
  assert cache[None] == 6

  # Graph mode
  with tf.Graph().as_default() as g:
    cache[None] += 5
    cache[g] += 3
  assert cache[g] == 8
  ```

  Example of a default factory with arguments:

  ```
  cache = ContextValueCache(lambda x: x + 1)
  g = tf.get_default_graph()

  # Example with keyword argument.
  value = cache.setdefault(key=g, kwargs={'x': 3})
  assert cache[g] == 4
  ```
  """

    def __init__(self, default_factory):
        self.default_factory = default_factory
        weakref.WeakKeyDictionary.__init__(self)

    def _key(self):
        if context.executing_eagerly():
            return _DUMMY_EAGER_GRAPH.key
        else:
            return ops.get_default_graph()

    def _get_parent_graph(self, graph):
        """Returns the parent graph or dummy eager object."""
        parent_graph = graph.outer_graph
        if not isinstance(parent_graph, func_graph.FuncGraph) and ops.executing_eagerly_outside_functions():
            return _DUMMY_EAGER_GRAPH.key
        return parent_graph

    def _get_recursive(self, key):
        """Gets the value at key or the closest parent graph."""
        value = self.get(key)
        if value is not None:
            return value
        if isinstance(key, func_graph.FuncGraph):
            return self._get_recursive(self._get_parent_graph(key))
        return None

    def __getitem__(self, key):
        """Gets the value at key (or current context), or sets default value.

    Args:
      key: May be `None` or `Graph`object. When `None`, the key is set to the
        current context.

    Returns:
      Either the cached or default value.
    """
        if key is None:
            key = self._key()
        value = self._get_recursive(key)
        if value is None:
            value = self[key] = self.default_factory()
        return value

    def setdefault(self, key=None, default=None, kwargs=None):
        """Sets the default value if key is not in dict, and returns the value."""
        if key is None:
            key = self._key()
        kwargs = kwargs or {}
        if default is None and key not in self:
            default = self.default_factory(**kwargs)
        return weakref.WeakKeyDictionary.setdefault(self, key, default)