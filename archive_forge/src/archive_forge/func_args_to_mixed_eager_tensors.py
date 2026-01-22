from google.protobuf import text_format
from tensorflow.core.framework import tensor_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import core
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
def args_to_mixed_eager_tensors(lists, ctx):
    """Converts a list of same-length lists of values to eager tensors."""
    del ctx
    assert len(lists) > 1
    lists_ret = [[]]
    for l in lists[1:]:
        if len(l) != len(lists[0]):
            raise ValueError('Expected list arguments to be the same length: %d != %d (%r vs. %r).' % (len(lists[0]), len(l), lists[0], l))
        lists_ret.append([])
    types = []
    for i in range(len(lists[0])):
        dtype = None
        for l in lists:
            if isinstance(l[i], core_types.Value):
                dtype = l[i].dtype
                break
        if dtype is None:
            lists_ret[0].append(tensor_conversion_registry.convert(lists[0][i]))
            dtype = lists_ret[0][i].dtype
            for j in range(1, len(lists)):
                lists_ret[j].append(tensor_conversion_registry.convert(lists[j][i], dtype=dtype))
        else:
            for j in range(len(lists)):
                lists_ret[j].append(tensor_conversion_registry.convert(lists[j][i], dtype=dtype))
        types.append(dtype.as_datatype_enum)
    return (types, lists_ret)