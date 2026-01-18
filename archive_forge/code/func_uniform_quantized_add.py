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
def uniform_quantized_add(lhs: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedAdd_T], rhs: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedAdd_T], lhs_scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], lhs_zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], rhs_scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], rhs_zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], output_scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], output_zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], lhs_quantization_min_val: int, lhs_quantization_max_val: int, rhs_quantization_min_val: int, rhs_quantization_max_val: int, output_quantization_min_val: int, output_quantization_max_val: int, lhs_quantization_axis: int=-1, rhs_quantization_axis: int=-1, output_quantization_axis: int=-1, name=None) -> _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedAdd_T]:
    """Perform quantized add of quantized Tensor `lhs` and quantized Tensor `rhs` to make quantized `output`.

  Given quantized `lhs` and quantized `rhs`, performs quantized add on `lhs` and `rhs` to make quantized `output`.

  `UniformQuantizedAdd` follows Numpy broadcasting rules.
  The two input array shapes are compared element-wise.
  Starting with the trailing dimensions, the two dimensions either have to be equal or one of them needs to be 1.

  `lhs` and `rhs` must be quantized Tensor, where data value is quantized using the formula:
  ```
  quantized_data = clip(original_data / scale + zero_point, quantization_min_val, quantization_max_val)
  ```
  `output` is also quantized, using the same formula.

  If `lhs` and `output` is both per-axis quantized, the quantization axis must match.
  Also, if `rhs` and `output` is both per-axis quantized, the quantization axis must match.
  *Match* means the axis must match when adding, regarding the broadcasting.
  i.e. For both operands `lhs` and `rhs`,
  if `operand.quantization_axis` >= 0 and `output.quantization_axis` >= 0,
  `operand.dims` - `operand.quantization_axis` must be equal to `output.dims` - `output.quantization_axis`.

  Args:
    lhs: A `Tensor`. Must be one of the following types: `qint32`.
      Must be a quantized tensor.
    rhs: A `Tensor`. Must have the same type as `lhs`.
      Must be a quantized tensor.
    lhs_scales: A `Tensor` of type `float32`.
      The float value(s) used as scale factors when quantizing the original data that `lhs` represents.
    lhs_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero points when quantizing original data that `lhs` represents.
      Must have same shape with `lhs_scales`.
    rhs_scales: A `Tensor` of type `float32`.
      The float value(s) used as scale factors when quantizing the original data that `rhs` represents.
    rhs_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero points when quantizing original data that `rhs` represents.
      Must have same shape with `rhs_scales`.
    output_scales: A `Tensor` of type `float32`.
      The float value(s) to use as scale factors when quantizing original data that `output` represents.
    output_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero points when quantizing original data that output represents.
      Must have same shape with `output_scales`.
    lhs_quantization_min_val: An `int`.
      The min value of the quantized data stored in `lhs`.
      For example, if `Tin` is `qint8`, this must be set to -127 if narrow range quantized or -128 if not.
    lhs_quantization_max_val: An `int`.
      The max value of the quantized data stored in `lhs`.
      For example, if `Tin` is `qint8`, this must be set to 127.
    rhs_quantization_min_val: An `int`.
      The min value of the quantized data stored in `rhs`.
      For example, if `Tin` is `qint8`, this must be set to -127 if narrow range quantized or -128 if not.
    rhs_quantization_max_val: An `int`.
      The max value of the quantized data stored in `rhs`.
      For example, if `Tin` is `qint8`, this must be set to 127.
    output_quantization_min_val: An `int`.
      The min value of the quantized data stored in `output`.
      For example, if  `Tout` is `qint8`, this must be set to -127 if narrow range quantized or -128 if not.
    output_quantization_max_val: An `int`.
      The max value of the quantized data stored in `output`.
      For example, if `Tout` is `qint8`, this must be set to 127.
    lhs_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For the `lhs`, only per-tensor quantization is supported.
      Thus, this must be set to -1.
      Other values will raise error at OpKernel construction.
    rhs_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For the `rhs`, only per-tensor quantization
      or per-channel quantization along `kernel_output_feature_dimension` is supported.
      Thus, this must be set to -1 or `dimension_numbers.kernel_output_feature_dimension`.
      Other values will raise error at OpKernel construction.
    output_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For the `output`, only per-tensor quantization or per-channel quantization along `output_feature_dimension` is supported.
      Thus, this must be set to -1 or `dimension_numbers.output_feature_dimension`.
      Other values will raise error at OpKernel construction.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `lhs`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'UniformQuantizedAdd', name, lhs, rhs, lhs_scales, lhs_zero_points, rhs_scales, rhs_zero_points, output_scales, output_zero_points, 'lhs_quantization_axis', lhs_quantization_axis, 'lhs_quantization_min_val', lhs_quantization_min_val, 'lhs_quantization_max_val', lhs_quantization_max_val, 'rhs_quantization_axis', rhs_quantization_axis, 'rhs_quantization_min_val', rhs_quantization_min_val, 'rhs_quantization_max_val', rhs_quantization_max_val, 'output_quantization_axis', output_quantization_axis, 'output_quantization_min_val', output_quantization_min_val, 'output_quantization_max_val', output_quantization_max_val)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return uniform_quantized_add_eager_fallback(lhs, rhs, lhs_scales, lhs_zero_points, rhs_scales, rhs_zero_points, output_scales, output_zero_points, lhs_quantization_axis=lhs_quantization_axis, lhs_quantization_min_val=lhs_quantization_min_val, lhs_quantization_max_val=lhs_quantization_max_val, rhs_quantization_axis=rhs_quantization_axis, rhs_quantization_min_val=rhs_quantization_min_val, rhs_quantization_max_val=rhs_quantization_max_val, output_quantization_axis=output_quantization_axis, output_quantization_min_val=output_quantization_min_val, output_quantization_max_val=output_quantization_max_val, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
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
    _, _, _op, _outputs = _op_def_library._apply_op_helper('UniformQuantizedAdd', lhs=lhs, rhs=rhs, lhs_scales=lhs_scales, lhs_zero_points=lhs_zero_points, rhs_scales=rhs_scales, rhs_zero_points=rhs_zero_points, output_scales=output_scales, output_zero_points=output_zero_points, lhs_quantization_min_val=lhs_quantization_min_val, lhs_quantization_max_val=lhs_quantization_max_val, rhs_quantization_min_val=rhs_quantization_min_val, rhs_quantization_max_val=rhs_quantization_max_val, output_quantization_min_val=output_quantization_min_val, output_quantization_max_val=output_quantization_max_val, lhs_quantization_axis=lhs_quantization_axis, rhs_quantization_axis=rhs_quantization_axis, output_quantization_axis=output_quantization_axis, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('lhs_quantization_axis', _op._get_attr_int('lhs_quantization_axis'), 'lhs_quantization_min_val', _op._get_attr_int('lhs_quantization_min_val'), 'lhs_quantization_max_val', _op._get_attr_int('lhs_quantization_max_val'), 'rhs_quantization_axis', _op._get_attr_int('rhs_quantization_axis'), 'rhs_quantization_min_val', _op._get_attr_int('rhs_quantization_min_val'), 'rhs_quantization_max_val', _op._get_attr_int('rhs_quantization_max_val'), 'output_quantization_axis', _op._get_attr_int('output_quantization_axis'), 'output_quantization_min_val', _op._get_attr_int('output_quantization_min_val'), 'output_quantization_max_val', _op._get_attr_int('output_quantization_max_val'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('UniformQuantizedAdd', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result