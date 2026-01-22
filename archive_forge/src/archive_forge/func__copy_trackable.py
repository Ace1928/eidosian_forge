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
def _copy_trackable(self, original_trackable):
    """Create a new instance for the input trackable.

    Args:
      original_trackable: The trackable instance to be copied.

    Raises:
      AttributeError: if the input trackable is not Variable or ShardedVariable.
    """
    if isinstance(original_trackable, ShardedVariable):
        self._copy_for_sharded_variable(original_trackable)
    elif isinstance(original_trackable, Variable):
        self._copy_for_variable(original_trackable)
    else:
        raise AttributeError('Only Variable or ShardedVariable can be copied.')