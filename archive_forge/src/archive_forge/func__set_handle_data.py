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
def _set_handle_data(list_handle, element_shape, element_dtype):
    """Sets type information on `list_handle` for consistency with graphs."""
    if isinstance(list_handle, ops.EagerTensor):
        if tensor_util.is_tf_type(element_shape):
            element_shape = tensor_shape.TensorShape(None)
        elif not isinstance(element_shape, tensor_shape.TensorShape):
            element_shape = tensor_shape.TensorShape(element_shape)
        handle_data = cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData()
        handle_data.is_set = True
        handle_data.shape_and_type.append(cpp_shape_inference_pb2.CppShapeInferenceResult.HandleShapeAndType(shape=element_shape.as_proto(), dtype=element_dtype.as_datatype_enum, type=full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_ARRAY)))
        list_handle._handle_data = handle_data