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
def __tf_experimental_restore_capture__(self, concrete_function, internal_capture):
    closure, spec = self.resource_handle_call_time_value()
    concrete_function.graph.replace_capture_with_deferred_capture(self._coordinator_instance.resource_handle, closure, spec, default_value=self._coordinator_instance.resource_handle, placeholder=internal_capture)
    return concrete_function.graph.deferred_external_captures[-1]