import functools
import threading
from tensorflow.python import tf2
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.trackable import base as tracking
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
class CallContextManager(object):
    """Context manager for `CallContext`."""

    def __init__(self, call_ctx, state):
        self._call_ctx = call_ctx
        self._state = state
        self._build_graph = state['build_graph']

    def __enter__(self):
        call_ctx = self._call_ctx
        self._prev_in_call = call_ctx.in_call
        self._prev_state = call_ctx._state
        call_ctx.in_call = True
        call_ctx._state = self._state
        if self._build_graph:
            self._prev_in_keras_graph = call_ctx._in_keras_graph
            call_ctx._in_keras_graph = call_ctx._in_keras_graph or getattr(backend.get_graph(), 'name', None) == 'keras_graph'

    def __exit__(self, *exc_info):
        call_ctx = self._call_ctx
        call_ctx.in_call = self._prev_in_call
        call_ctx._state = self._prev_state
        if self._build_graph:
            call_ctx._in_keras_graph = self._prev_in_keras_graph