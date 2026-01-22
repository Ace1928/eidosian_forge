import abc
import collections
from tensorflow.python.eager import context
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.linalg.sparse import gen_sparse_csr_matrix_ops as sm_ops
from tensorflow.python.ops.linalg.sparse.gen_sparse_csr_matrix_ops import *
def _create_handle_data_proto(shape_proto, dtype_enum):
    """Create handle data based on shape and dtype protos."""
    variant_shape_and_type_data = cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData()
    variant_shape_and_type_data.is_set = True
    variant_shape_and_type_data.shape_and_type.extend([cpp_shape_inference_pb2.CppShapeInferenceResult.HandleShapeAndType(shape=shape_proto, dtype=dtype_enum)])
    return variant_shape_and_type_data