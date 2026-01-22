import collections
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python import tf2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import internal
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util.tf_export import tf_export
def _numpy(self):
    """Returns a numpy `array` with the values for this `SparseTensor`.

    Requires that this `SparseTensor` was constructed in eager execution mode.
    """
    if not self._is_eager():
        raise ValueError('SparseTensor.numpy() is only supported in eager mode.')
    arr = np.zeros(self.dense_shape, dtype=self.dtype.as_numpy_dtype())
    for i, v in zip(self.indices, self.values):
        arr[tuple(i)] = v
    return arr