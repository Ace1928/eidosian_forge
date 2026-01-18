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
def unpack_eager_fallback(value: _atypes.TensorFuzzingAnnotation[TV_Unpack_T], num: int, axis: int, name, ctx):
    num = _execute.make_int(num, 'num')
    if axis is None:
        axis = 0
    axis = _execute.make_int(axis, 'axis')
    _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [])
    _inputs_flat = [value]
    _attrs = ('num', num, 'T', _attr_T, 'axis', axis)
    _result = _execute.execute(b'Unpack', num, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('Unpack', _inputs_flat, _attrs, _result)
    return _result