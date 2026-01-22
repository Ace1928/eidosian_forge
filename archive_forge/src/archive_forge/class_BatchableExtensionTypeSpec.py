import abc
import typing
import warnings
import typing_extensions
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type_field
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import composite_tensor_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
class BatchableExtensionTypeSpec(ExtensionTypeSpec, type_spec.BatchableTypeSpec):
    """Base class for TypeSpecs for BatchableExtensionTypes."""
    __batch_encoder__ = ExtensionTypeBatchEncoder()

    def _batch(self, batch_size):
        return self.__batch_encoder__.batch(self, batch_size)

    def _unbatch(self):
        return self.__batch_encoder__.unbatch(self)

    def _to_tensor_list(self, value):
        return type_spec.batchable_to_tensor_list(self, value)

    def _to_batched_tensor_list(self, value):
        return type_spec.batchable_to_tensor_list(self, value, minimum_rank=1)

    def _from_compatible_tensor_list(self, tensor_list):
        return type_spec.batchable_from_tensor_list(self, tensor_list)

    @property
    def _flat_tensor_specs(self):
        return type_spec.get_batchable_flat_tensor_specs(self)