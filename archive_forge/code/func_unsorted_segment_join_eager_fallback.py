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
def unsorted_segment_join_eager_fallback(inputs: _atypes.TensorFuzzingAnnotation[_atypes.String], segment_ids: _atypes.TensorFuzzingAnnotation[TV_UnsortedSegmentJoin_Tindices], num_segments: _atypes.TensorFuzzingAnnotation[TV_UnsortedSegmentJoin_Tnumsegments], separator: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    if separator is None:
        separator = ''
    separator = _execute.make_str(separator, 'separator')
    _attr_Tindices, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64])
    _attr_Tnumsegments, (num_segments,) = _execute.args_to_matching_eager([num_segments], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    inputs = _ops.convert_to_tensor(inputs, _dtypes.string)
    _inputs_flat = [inputs, segment_ids, num_segments]
    _attrs = ('separator', separator, 'Tindices', _attr_Tindices, 'Tnumsegments', _attr_Tnumsegments)
    _result = _execute.execute(b'UnsortedSegmentJoin', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('UnsortedSegmentJoin', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result