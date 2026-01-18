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
def unicode_encode_eager_fallback(input_values: _atypes.TensorFuzzingAnnotation[_atypes.Int32], input_splits: _atypes.TensorFuzzingAnnotation[TV_UnicodeEncode_Tsplits], output_encoding: str, errors: str, replacement_char: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    output_encoding = _execute.make_str(output_encoding, 'output_encoding')
    if errors is None:
        errors = 'replace'
    errors = _execute.make_str(errors, 'errors')
    if replacement_char is None:
        replacement_char = 65533
    replacement_char = _execute.make_int(replacement_char, 'replacement_char')
    _attr_Tsplits, (input_splits,) = _execute.args_to_matching_eager([input_splits], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int64)
    input_values = _ops.convert_to_tensor(input_values, _dtypes.int32)
    _inputs_flat = [input_values, input_splits]
    _attrs = ('errors', errors, 'output_encoding', output_encoding, 'replacement_char', replacement_char, 'Tsplits', _attr_Tsplits)
    _result = _execute.execute(b'UnicodeEncode', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('UnicodeEncode', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result