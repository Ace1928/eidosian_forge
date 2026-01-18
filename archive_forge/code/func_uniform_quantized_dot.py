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
def uniform_quantized_dot(lhs: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedDot_Tin], rhs: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedDot_Tin], lhs_scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], lhs_zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], rhs_scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], rhs_zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], output_scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], output_zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], Tout: TV_UniformQuantizedDot_Tout, lhs_quantization_min_val: int, lhs_quantization_max_val: int, rhs_quantization_min_val: int, rhs_quantization_max_val: int, output_quantization_min_val: int, output_quantization_max_val: int, lhs_quantization_axis: int=-1, rhs_quantization_axis: int=-1, output_quantization_axis: int=-1, name=None) -> _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedDot_Tout]:
    """Perform quantized dot of quantized Tensor `lhs` and quantized Tensor `rhs` to make quantized `output`.

  Given quantized `lhs` and quantized `rhs`, performs quantized dot on `lhs` and `rhs` to make quantized `output`.
  `lhs` and `rhs` must be 2D Tensors and the lhs.dim_size(1) must match rhs.dim_size(0).
  `lhs` and `rhs` must be quantized Tensor, where data value is quantized using the formula:
  quantized_data = clip(original_data / scale + zero_point, quantization_min_val, quantization_max_val).
  `output` is also quantized, using the same formula.
  If `rhs` is per-tensor quantized, `output` must be also per-tensor quantized.

  Args:
    lhs: A `Tensor`. Must be one of the following types: `qint8`.
      Must be a 2D Tensor of Tin.
    rhs: A `Tensor`. Must have the same type as `lhs`.
      Must be a 2D Tensor of Tin.
    lhs_scales: A `Tensor` of type `float32`.
      The float value(s) used as scale when quantizing original data that lhs represents.
      Must be a scalar Tensor (lhs supports only per-tensor quantization).
    lhs_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point when quantizing original data that lhs represents.
      Same shape condition as lhs_scales.
    rhs_scales: A `Tensor` of type `float32`.
      The float value(s) used as scale when quantizing original data that rhs represents.
      Must be a scalar Tensor (per-tensor quantization) or 1D Tensor of size (rhs.dim_size(1),) (per-channel quantization).
    rhs_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point when quantizing original data that rhs represents.
      Same shape condition as rhs_scales.
    output_scales: A `Tensor` of type `float32`.
      The float value(s) to use as scales when quantizing original data that output represents.
      Must be a scalar Tensor (per-tensor quantization) or 1D Tensor of size (output.dim_size(1),) (per-channel quantization).
      If rhs is per-tensor quantized, output must be also per-tensor quantized.
      This means that if rhs_scales and rhs_zero_points are scalar Tensors, output_scales and output_zero_points must be scalar Tensors as well.
    output_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point when quantizing original data that output represents.
      Same shape condition as rhs_scales.
    Tout: A `tf.DType` from: `tf.qint32`. The type of output Tensor.
    lhs_quantization_min_val: An `int`.
      The min value of the quantized data stored in lhs.
      For example, if Tin is qint8, this must be set to -127 if narrow range quantized or -128 if not.
    lhs_quantization_max_val: An `int`.
      The max value of the quantized data stored in rhs.
      For example, if Tin is qint8, this must be set to 127.
    rhs_quantization_min_val: An `int`.
      The min value of the quantized data stored in rhs.
      For example, if Trhs is qint8, this must be set to -127 if narrow range quantized or -128 if not.
    rhs_quantization_max_val: An `int`.
      The max value of the quantized data stored in rhs.
      For example, if Trhs is qint8, this must be set to 127.
    output_quantization_min_val: An `int`.
      The min value of the quantized data stored in output.
      For example, if Tout is qint8, this must be set to -127 if narrow range quantized or -128 if not.
    output_quantization_max_val: An `int`.
      The max value of the quantized data stored in output.
      For example, if Tout is qint8, this must be set to 127.
    lhs_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For dot op lhs, only per-tensor quantization is supported.
      Thus, this attribute must be set to -1. Other values are rejected.
    rhs_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For dot op rhs, only per-tensor quantization or per-channel quantization along dimension 1 is supported.
      Thus, this attribute must be set to -1 or 1. Other values are rejected.
    output_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For dot op output, only per-tensor quantization or per-channel quantization along dimension 1 is supported.
      Thus, this attribute must be set to -1 or 1. Other values are rejected.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'UniformQuantizedDot', name, lhs, rhs, lhs_scales, lhs_zero_points, rhs_scales, rhs_zero_points, output_scales, output_zero_points, 'Tout', Tout, 'lhs_quantization_axis', lhs_quantization_axis, 'lhs_quantization_min_val', lhs_quantization_min_val, 'lhs_quantization_max_val', lhs_quantization_max_val, 'rhs_quantization_axis', rhs_quantization_axis, 'rhs_quantization_min_val', rhs_quantization_min_val, 'rhs_quantization_max_val', rhs_quantization_max_val, 'output_quantization_axis', output_quantization_axis, 'output_quantization_min_val', output_quantization_min_val, 'output_quantization_max_val', output_quantization_max_val)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return uniform_quantized_dot_eager_fallback(lhs, rhs, lhs_scales, lhs_zero_points, rhs_scales, rhs_zero_points, output_scales, output_zero_points, Tout=Tout, lhs_quantization_axis=lhs_quantization_axis, lhs_quantization_min_val=lhs_quantization_min_val, lhs_quantization_max_val=lhs_quantization_max_val, rhs_quantization_axis=rhs_quantization_axis, rhs_quantization_min_val=rhs_quantization_min_val, rhs_quantization_max_val=rhs_quantization_max_val, output_quantization_axis=output_quantization_axis, output_quantization_min_val=output_quantization_min_val, output_quantization_max_val=output_quantization_max_val, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
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
    _, _, _op, _outputs = _op_def_library._apply_op_helper('UniformQuantizedDot', lhs=lhs, rhs=rhs, lhs_scales=lhs_scales, lhs_zero_points=lhs_zero_points, rhs_scales=rhs_scales, rhs_zero_points=rhs_zero_points, output_scales=output_scales, output_zero_points=output_zero_points, Tout=Tout, lhs_quantization_min_val=lhs_quantization_min_val, lhs_quantization_max_val=lhs_quantization_max_val, rhs_quantization_min_val=rhs_quantization_min_val, rhs_quantization_max_val=rhs_quantization_max_val, output_quantization_min_val=output_quantization_min_val, output_quantization_max_val=output_quantization_max_val, lhs_quantization_axis=lhs_quantization_axis, rhs_quantization_axis=rhs_quantization_axis, output_quantization_axis=output_quantization_axis, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Tin', _op._get_attr_type('Tin'), 'Tout', _op._get_attr_type('Tout'), 'lhs_quantization_axis', _op._get_attr_int('lhs_quantization_axis'), 'lhs_quantization_min_val', _op._get_attr_int('lhs_quantization_min_val'), 'lhs_quantization_max_val', _op._get_attr_int('lhs_quantization_max_val'), 'rhs_quantization_axis', _op._get_attr_int('rhs_quantization_axis'), 'rhs_quantization_min_val', _op._get_attr_int('rhs_quantization_min_val'), 'rhs_quantization_max_val', _op._get_attr_int('rhs_quantization_max_val'), 'output_quantization_axis', _op._get_attr_int('output_quantization_axis'), 'output_quantization_min_val', _op._get_attr_int('output_quantization_min_val'), 'output_quantization_max_val', _op._get_attr_int('output_quantization_max_val'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('UniformQuantizedDot', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result