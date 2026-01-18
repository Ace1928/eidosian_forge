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
def xla_dynamic_update_slice_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_XlaDynamicUpdateSlice_T], update: _atypes.TensorFuzzingAnnotation[TV_XlaDynamicUpdateSlice_T], indices: _atypes.TensorFuzzingAnnotation[TV_XlaDynamicUpdateSlice_Tindices], name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_XlaDynamicUpdateSlice_T]:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([input, update], ctx, [])
    input, update = _inputs_T
    _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64])
    _inputs_flat = [input, update, indices]
    _attrs = ('T', _attr_T, 'Tindices', _attr_Tindices)
    _result = _execute.execute(b'XlaDynamicUpdateSlice', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('XlaDynamicUpdateSlice', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result