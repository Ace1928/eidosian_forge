from tensorflow.core.config import flags
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
def IsTrainable(tensor_or_dtype):
    """Determines whether a tensor or dtype supports infinitesimal changes."""
    if tensor_util.is_tf_type(tensor_or_dtype):
        dtype = _DTypeFromTensor(tensor_or_dtype)
    else:
        dtype = tensor_or_dtype
    dtype = dtypes.as_dtype(dtype)
    trainable_dtypes = [dtypes.float16, dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128, dtypes.resource, dtypes.variant, dtypes.bfloat16]
    if flags.config().enable_quantized_dtypes_training.value():
        trainable_dtypes.extend([dtypes.qint8, dtypes.qint16, dtypes.qint32, dtypes.quint8, dtypes.quint16])
    return dtype.base_dtype in trainable_dtypes