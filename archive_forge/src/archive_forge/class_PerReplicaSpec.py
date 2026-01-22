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
class PerReplicaSpec(type_spec.TypeSpec):
    """Type specification for a `PerReplica`."""
    __slots__ = ['_value_specs']
    value_type = property(lambda self: PerReplica)

    def __init__(self, *value_specs):
        self._value_specs = tuple(value_specs)

    def _serialize(self):
        return self._value_specs

    @property
    def _component_specs(self):
        return self._value_specs

    def _to_components(self, value):
        replica_context = distribute_lib.get_replica_context()
        if replica_context is not None and replica_context.num_replicas_in_sync > 1:
            raise ValueError('Flattening a PerReplica to components is not supported in replica context.')
        return value._values

    def _from_components(self, tensor_list):
        return PerReplica(tensor_list)