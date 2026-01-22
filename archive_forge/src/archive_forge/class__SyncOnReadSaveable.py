import copy
from typing import Optional
import weakref
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.types import core
from tensorflow.python.types import distribute as ds_types
from tensorflow.python.types import trace
class _SyncOnReadSaveable(saveable_object.SaveableObject):
    """Class for defining how to restore a SyncOnReadVariable."""

    def __init__(self, sync_on_read_variable, name):
        self._sync_on_read_variable = sync_on_read_variable
        tensor, spec = values_util.get_on_read_saveable(sync_on_read_variable, sync_on_read_variable._primary, name)
        super(_SyncOnReadSaveable, self).__init__(tensor, spec, name)

    def restore(self, restored_tensors, restored_shapes):
        """Restore the same value into all variables."""
        tensor, = restored_tensors
        return values_util.get_on_read_restore_ops(self._sync_on_read_variable, tensor, self._sync_on_read_variable.aggregation)