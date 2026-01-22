import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
def _static_check(self):
    """Checks if the object is internally consistent.

    Raises:
      ValueError if inconsistent.
    """
    my_dtype = self.dtype
    if self._uniform_row_length is not None:
        if self._uniform_row_length.dtype != my_dtype:
            raise ValueError('_uniform_row_length.dtype=' + str(self._uniform_row_length.dtype) + ', not ' + str(my_dtype))
    if self._row_lengths is not None and self._row_lengths.dtype != my_dtype:
        raise ValueError('_row_lengths.dtype=' + str(self._row_lengths.dtype) + ', not ' + str(my_dtype))
    if self._value_rowids is not None and self._value_rowids.dtype != my_dtype:
        raise ValueError('_value_rowids.dtype=' + str(self._value_rowids.dtype) + ', not ' + str(my_dtype))
    if self._nrows is not None and self._nrows.dtype != my_dtype:
        raise ValueError('_nrows.dtype=' + str(self._nrows.dtype) + ', not ' + str(my_dtype))