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
def xla_conv_v2_eager_fallback(lhs: _atypes.TensorFuzzingAnnotation[TV_XlaConvV2_LhsT], rhs: _atypes.TensorFuzzingAnnotation[TV_XlaConvV2_RhsT], window_strides: _atypes.TensorFuzzingAnnotation[TV_XlaConvV2_Tindices], padding: _atypes.TensorFuzzingAnnotation[TV_XlaConvV2_Tindices], lhs_dilation: _atypes.TensorFuzzingAnnotation[TV_XlaConvV2_Tindices], rhs_dilation: _atypes.TensorFuzzingAnnotation[TV_XlaConvV2_Tindices], feature_group_count: _atypes.TensorFuzzingAnnotation[TV_XlaConvV2_Tindices], dimension_numbers: str, precision_config: str, preferred_element_type: TV_XlaConvV2_preferred_element_type, batch_group_count: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_XlaConvV2_preferred_element_type]:
    dimension_numbers = _execute.make_str(dimension_numbers, 'dimension_numbers')
    precision_config = _execute.make_str(precision_config, 'precision_config')
    preferred_element_type = _execute.make_type(preferred_element_type, 'preferred_element_type')
    if batch_group_count is None:
        batch_group_count = 1
    batch_group_count = _execute.make_int(batch_group_count, 'batch_group_count')
    _attr_LhsT, (lhs,) = _execute.args_to_matching_eager([lhs], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64])
    _attr_RhsT, (rhs,) = _execute.args_to_matching_eager([rhs], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64])
    _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count], ctx, [_dtypes.int32, _dtypes.int64])
    window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count = _inputs_Tindices
    _inputs_flat = [lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count]
    _attrs = ('LhsT', _attr_LhsT, 'RhsT', _attr_RhsT, 'Tindices', _attr_Tindices, 'dimension_numbers', dimension_numbers, 'precision_config', precision_config, 'preferred_element_type', preferred_element_type, 'batch_group_count', batch_group_count)
    _result = _execute.execute(b'XlaConvV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('XlaConvV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result