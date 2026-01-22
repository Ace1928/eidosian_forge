import contextlib
import copy
import weakref
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.trackable import base
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
@_destruction_context.setter
def _destruction_context(self, destruction_context):
    self._self_destruction_context = destruction_context