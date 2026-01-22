from google.protobuf import text_format
from tensorflow.core.framework import tensor_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import core
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
def args_to_matching_eager(l, ctx, allowed_dtypes, default_dtype=None):
    """Convert sequence `l` to eager same-type Tensors."""
    del ctx
    if not l and default_dtype is not None:
        return (default_dtype, [])
    for x in l:
        if not isinstance(x, core_types.Value):
            break
    else:
        return (l[0]._datatype_enum(), l)
    dtype = None
    for t in l:
        if isinstance(t, core_types.Value):
            dtype = t.dtype
            break
    if dtype is None:
        ret = []
        for t in l:
            tensor = None
            if dtype is None and allowed_dtypes:
                tensor = tensor_conversion_registry.convert(t)
                if tensor.dtype not in allowed_dtypes:
                    tensor = None
            if tensor is None:
                tensor = tensor_conversion_registry.convert(t, dtype, preferred_dtype=default_dtype)
            ret.append(tensor)
            if dtype is None:
                dtype = tensor.dtype
    else:
        ret = [tensor_conversion_registry.convert(t, dtype) for t in l]
    keras_symbolic_tensors = [x for x in ret if _is_keras_symbolic_tensor(x)]
    if keras_symbolic_tensors:
        raise core._SymbolicException('Using symbolic output of a Keras layer during eager execution {}'.format(keras_symbolic_tensors))
    return (dtype.as_datatype_enum, ret)