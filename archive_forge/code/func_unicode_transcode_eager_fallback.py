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
def unicode_transcode_eager_fallback(input: _atypes.TensorFuzzingAnnotation[_atypes.String], input_encoding: str, output_encoding: str, errors: str, replacement_char: int, replace_control_characters: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    input_encoding = _execute.make_str(input_encoding, 'input_encoding')
    output_encoding = _execute.make_str(output_encoding, 'output_encoding')
    if errors is None:
        errors = 'replace'
    errors = _execute.make_str(errors, 'errors')
    if replacement_char is None:
        replacement_char = 65533
    replacement_char = _execute.make_int(replacement_char, 'replacement_char')
    if replace_control_characters is None:
        replace_control_characters = False
    replace_control_characters = _execute.make_bool(replace_control_characters, 'replace_control_characters')
    input = _ops.convert_to_tensor(input, _dtypes.string)
    _inputs_flat = [input]
    _attrs = ('input_encoding', input_encoding, 'output_encoding', output_encoding, 'errors', errors, 'replacement_char', replacement_char, 'replace_control_characters', replace_control_characters)
    _result = _execute.execute(b'UnicodeTranscode', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('UnicodeTranscode', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result