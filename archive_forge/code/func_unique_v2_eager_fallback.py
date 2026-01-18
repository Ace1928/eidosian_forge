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
def unique_v2_eager_fallback(x: _atypes.TensorFuzzingAnnotation[TV_UniqueV2_T], axis: _atypes.TensorFuzzingAnnotation[TV_UniqueV2_Taxis], out_idx: TV_UniqueV2_out_idx, name, ctx):
    if out_idx is None:
        out_idx = _dtypes.int32
    out_idx = _execute.make_type(out_idx, 'out_idx')
    _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [])
    _attr_Taxis, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int64)
    _inputs_flat = [x, axis]
    _attrs = ('T', _attr_T, 'Taxis', _attr_Taxis, 'out_idx', out_idx)
    _result = _execute.execute(b'UniqueV2', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('UniqueV2', _inputs_flat, _attrs, _result)
    _result = _UniqueV2Output._make(_result)
    return _result