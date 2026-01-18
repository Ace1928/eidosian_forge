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
def xla_dequantize_eager_fallback(input: _atypes.TensorFuzzingAnnotation[_atypes.UInt32], min_range: float, max_range: float, mode: str, transpose_output: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.BFloat16]:
    min_range = _execute.make_float(min_range, 'min_range')
    max_range = _execute.make_float(max_range, 'max_range')
    mode = _execute.make_str(mode, 'mode')
    transpose_output = _execute.make_bool(transpose_output, 'transpose_output')
    input = _ops.convert_to_tensor(input, _dtypes.uint32)
    _inputs_flat = [input]
    _attrs = ('min_range', min_range, 'max_range', max_range, 'mode', mode, 'transpose_output', transpose_output)
    _result = _execute.execute(b'XlaDequantize', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('XlaDequantize', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result