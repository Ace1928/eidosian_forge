import contextlib
import copy
import functools
import threading
import weakref
import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
class LocalResourceRestoreContext(object):
    """Class holding information of a distributed instance, e.g. StaticHashTable.

  Pairing use with context manager `with_local_resource_restore_context` allows
  operations under this context manager to conveniently gets information of a
  component of the `RestoredDistributedTable` (and other restored distributed
  `CapturableResource` if we're supporting their distribution in the future),
  instead of looking it up from the mapping of the worker-to-resource handle.
  This is especially useful when we know which instance the operations should
  execute with and the mapping is not available yet.
  """

    def __init__(self, instance):
        self.instance = instance