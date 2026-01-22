import numpy
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base as trackable
@classmethod
def _deserialize_from_proto(cls, object_proto, operation_attributes, **kwargs):
    tensor_proto = operation_attributes[object_proto.constant.operation]['value'].tensor
    ndarray = tensor_util.MakeNdarray(tensor_proto)
    if dtypes.as_dtype(tensor_proto.dtype) == dtypes.string:
        with ops.device('CPU'):
            imported_constant = constant_op.constant(ndarray)
    else:
        imported_constant = constant_op.constant(ndarray)
    return imported_constant