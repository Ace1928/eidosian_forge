import inspect
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
def _tf_int(x, base):
    if base not in (10, UNSPECIFIED):
        raise NotImplementedError('base {} not supported for int'.format(base))
    if x.dtype == dtypes.string:
        return gen_parsing_ops.string_to_number(x, out_type=dtypes.int32)
    return math_ops.cast(x, dtype=dtypes.int32)