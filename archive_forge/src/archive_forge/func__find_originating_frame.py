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
def _find_originating_frame(caller_fn_scope, innermost=True):
    """Locates the frame in which `caller_fn_scope` was defined."""
    ctx_frame = inspect.currentframe()
    result = None
    while ctx_frame is not None:
        if ctx_frame.f_locals.get(caller_fn_scope.name, None) is caller_fn_scope:
            result = ctx_frame
            if innermost:
                break
        ctx_frame = ctx_frame.f_back
    assert result is not None, 'the conversion process should ensure the caller_fn_scope is always found somewhere on the call stack'
    return result