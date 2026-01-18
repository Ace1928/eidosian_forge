import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def xla_concat_nd_eager_fallback(inputs: List[_atypes.TensorFuzzingAnnotation[TV_XlaConcatND_T]], num_concats, paddings, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_XlaConcatND_T]:
    if not isinstance(inputs, (list, tuple)):
        raise TypeError("Expected list for 'inputs' argument to 'xla_concat_nd' Op, not %r." % inputs)
    _attr_N = len(inputs)
    if not isinstance(num_concats, (list, tuple)):
        raise TypeError("Expected list for 'num_concats' argument to 'xla_concat_nd' Op, not %r." % num_concats)
    num_concats = [_execute.make_int(_i, 'num_concats') for _i in num_concats]
    if paddings is None:
        paddings = []
    if not isinstance(paddings, (list, tuple)):
        raise TypeError("Expected list for 'paddings' argument to 'xla_concat_nd' Op, not %r." % paddings)
    paddings = [_execute.make_int(_i, 'paddings') for _i in paddings]
    _attr_T, inputs = _execute.args_to_matching_eager(list(inputs), ctx, [])
    _inputs_flat = list(inputs)
    _attrs = ('T', _attr_T, 'N', _attr_N, 'num_concats', num_concats, 'paddings', paddings)
    _result = _execute.execute(b'XlaConcatND', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('XlaConcatND', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result