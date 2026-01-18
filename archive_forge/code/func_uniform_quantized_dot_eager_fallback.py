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
def uniform_quantized_dot_eager_fallback(lhs: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedDot_Tin], rhs: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedDot_Tin], lhs_scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], lhs_zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], rhs_scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], rhs_zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], output_scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], output_zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], Tout: TV_UniformQuantizedDot_Tout, lhs_quantization_min_val: int, lhs_quantization_max_val: int, rhs_quantization_min_val: int, rhs_quantization_max_val: int, output_quantization_min_val: int, output_quantization_max_val: int, lhs_quantization_axis: int, rhs_quantization_axis: int, output_quantization_axis: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedDot_Tout]:
    Tout = _execute.make_type(Tout, 'Tout')
    lhs_quantization_min_val = _execute.make_int(lhs_quantization_min_val, 'lhs_quantization_min_val')
    lhs_quantization_max_val = _execute.make_int(lhs_quantization_max_val, 'lhs_quantization_max_val')
    rhs_quantization_min_val = _execute.make_int(rhs_quantization_min_val, 'rhs_quantization_min_val')
    rhs_quantization_max_val = _execute.make_int(rhs_quantization_max_val, 'rhs_quantization_max_val')
    output_quantization_min_val = _execute.make_int(output_quantization_min_val, 'output_quantization_min_val')
    output_quantization_max_val = _execute.make_int(output_quantization_max_val, 'output_quantization_max_val')
    if lhs_quantization_axis is None:
        lhs_quantization_axis = -1
    lhs_quantization_axis = _execute.make_int(lhs_quantization_axis, 'lhs_quantization_axis')
    if rhs_quantization_axis is None:
        rhs_quantization_axis = -1
    rhs_quantization_axis = _execute.make_int(rhs_quantization_axis, 'rhs_quantization_axis')
    if output_quantization_axis is None:
        output_quantization_axis = -1
    output_quantization_axis = _execute.make_int(output_quantization_axis, 'output_quantization_axis')
    _attr_Tin, _inputs_Tin = _execute.args_to_matching_eager([lhs, rhs], ctx, [_dtypes.qint8])
    lhs, rhs = _inputs_Tin
    lhs_scales = _ops.convert_to_tensor(lhs_scales, _dtypes.float32)
    lhs_zero_points = _ops.convert_to_tensor(lhs_zero_points, _dtypes.int32)
    rhs_scales = _ops.convert_to_tensor(rhs_scales, _dtypes.float32)
    rhs_zero_points = _ops.convert_to_tensor(rhs_zero_points, _dtypes.int32)
    output_scales = _ops.convert_to_tensor(output_scales, _dtypes.float32)
    output_zero_points = _ops.convert_to_tensor(output_zero_points, _dtypes.int32)
    _inputs_flat = [lhs, rhs, lhs_scales, lhs_zero_points, rhs_scales, rhs_zero_points, output_scales, output_zero_points]
    _attrs = ('Tin', _attr_Tin, 'Tout', Tout, 'lhs_quantization_axis', lhs_quantization_axis, 'lhs_quantization_min_val', lhs_quantization_min_val, 'lhs_quantization_max_val', lhs_quantization_max_val, 'rhs_quantization_axis', rhs_quantization_axis, 'rhs_quantization_min_val', rhs_quantization_min_val, 'rhs_quantization_max_val', rhs_quantization_max_val, 'output_quantization_axis', output_quantization_axis, 'output_quantization_min_val', output_quantization_min_val, 'output_quantization_max_val', output_quantization_max_val)
    _result = _execute.execute(b'UniformQuantizedDot', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('UniformQuantizedDot', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result