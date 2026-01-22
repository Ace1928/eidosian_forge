from typing import Optional, Type
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import handle_data_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.types import internal
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
class DenseSpec(type_spec.TypeSpec):
    """Describes a dense object with shape, dtype, and name."""
    __slots__ = ['_shape', '_dtype', '_name']
    _component_specs = property(lambda self: self)

    def __init__(self, shape, dtype=dtypes.float32, name=None):
        """Creates a TensorSpec.

    Args:
      shape: Value convertible to `tf.TensorShape`. The shape of the tensor.
      dtype: Value convertible to `tf.DType`. The type of the tensor values.
      name: Optional name for the Tensor.

    Raises:
      TypeError: If shape is not convertible to a `tf.TensorShape`, or dtype is
        not convertible to a `tf.DType`.
    """
        self._shape = tensor_shape.TensorShape(shape)
        self._dtype = dtypes.as_dtype(dtype)
        self._name = name

    @property
    def shape(self):
        """Returns the `TensorShape` that represents the shape of the tensor."""
        return self._shape

    @property
    def dtype(self):
        """Returns the `dtype` of elements in the tensor."""
        return self._dtype

    @property
    def name(self):
        """Returns the (optionally provided) name of the described tensor."""
        return self._name

    def is_compatible_with(self, spec_or_value):
        return isinstance(spec_or_value, (DenseSpec, self.value_type)) and self._dtype.is_compatible_with(spec_or_value.dtype) and self._shape.is_compatible_with(spec_or_value.shape)

    def __repr__(self):
        return '{}(shape={}, dtype={}, name={})'.format(type(self).__name__, self.shape, repr(self.dtype), repr(self.name))

    def __hash__(self):
        return hash((self._shape, self.dtype))

    def __eq__(self, other):
        return type(self) is type(other) and self._shape == other._shape and (self._dtype == other._dtype) and (self._name == other._name)

    def __ne__(self, other):
        return not self == other

    def _serialize(self):
        return (self._shape, self._dtype, self._name)

    def _to_legacy_output_types(self):
        return self._dtype

    def _to_legacy_output_shapes(self):
        return self._shape

    def _to_legacy_output_classes(self):
        return self.value_type