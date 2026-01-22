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
@ops.RegisterGradient('TensorListFromTensor')
def _TensorListFromTensorGrad(op, dlist):
    """Gradient for TensorListFromTensor."""
    t = op.inputs[0]
    if t.shape.dims and t.shape.dims[0].value is not None:
        num_elements = t.shape.dims[0].value
    else:
        num_elements = None
    if dlist is None:
        dlist = empty_tensor_list(element_dtype=t.dtype, element_shape=gen_list_ops.tensor_list_element_shape(op.outputs[0], shape_type=dtypes.int32))
    tensor_grad = gen_list_ops.tensor_list_stack(dlist, element_shape=array_ops.slice(array_ops.shape(t), [1], [-1]), element_dtype=t.dtype, num_elements=num_elements)
    shape_grad = None
    return (tensor_grad, shape_grad)