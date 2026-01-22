import atexit
import collections
import copy
import queue
import threading
import time
import weakref
from absl import logging
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute.sharded_variable import ShardedVariable
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.ops.resource_variable_ops import UninitializedVariable
from tensorflow.python.ops.variables import Variable
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.util import object_identity
def _copy_for_sharded_variable(self, original_var):
    """Create a new instance for the input ShardedVariable.

    Args:
      original_var: Input ShardedVariable object to be copied.
    """
    copied_vars = []
    for v in original_var._variables:
        self._copy_for_variable(v)
        copied_vars.append(self._object_map[v])
    self._object_map[original_var] = ShardedVariable(copied_vars, name=original_var.name)