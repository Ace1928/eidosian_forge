import numpy as np
from tensorflow.core.framework import full_type_pb2
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops.gen_list_ops import *
@ops.RegisterGradient('TensorListScatter')
@ops.RegisterGradient('TensorListScatterV2')
def _TensorListScatterGrad(op, dlist):
    """Gradient function for TensorListScatter."""
    tensor = op.inputs[0]
    indices = op.inputs[1]
    dtensor = gen_list_ops.tensor_list_gather(dlist, indices, element_shape=array_ops.slice(array_ops.shape(tensor), [1], [-1]), element_dtype=tensor.dtype)
    if op.type == 'TensorListScatterV2':
        return (dtensor, None, None, None)
    else:
        return (dtensor, None, None)