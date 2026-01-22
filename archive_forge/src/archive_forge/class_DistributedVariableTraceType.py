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
class DistributedVariableTraceType(trace.TraceType):
    """TraceType of DistributedVariable objects."""

    def __init__(self, distributed_variable):
        self.distributed_variable = distributed_variable
        self.components = (tuple(distributed_variable.shape.as_list()), distributed_variable.dtype)

    def is_subtype_of(self, other):
        return self == other

    def most_specific_common_supertype(self, others):
        return self if all((self == other for other in others)) else None

    def placeholder_value(self, placeholder_context=None):
        return self.distributed_variable

    def _to_tensors(self, value):
        return []

    def _cast(self, value, _):
        return value

    def __hash__(self) -> int:
        return hash(self.components)

    def __eq__(self, other) -> bool:
        if not isinstance(other, DistributedVariableTraceType):
            return False
        return self.components == other.components