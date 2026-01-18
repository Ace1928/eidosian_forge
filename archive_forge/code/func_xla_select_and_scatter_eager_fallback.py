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
def xla_select_and_scatter_eager_fallback(operand: _atypes.TensorFuzzingAnnotation[TV_XlaSelectAndScatter_T], window_dimensions: _atypes.TensorFuzzingAnnotation[TV_XlaSelectAndScatter_Tindices], window_strides: _atypes.TensorFuzzingAnnotation[TV_XlaSelectAndScatter_Tindices], padding: _atypes.TensorFuzzingAnnotation[TV_XlaSelectAndScatter_Tindices], source: _atypes.TensorFuzzingAnnotation[TV_XlaSelectAndScatter_T], init_value: _atypes.TensorFuzzingAnnotation[TV_XlaSelectAndScatter_T], select, scatter, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_XlaSelectAndScatter_T]:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([operand, source, init_value], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64])
    operand, source, init_value = _inputs_T
    _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([window_dimensions, window_strides, padding], ctx, [_dtypes.int32, _dtypes.int64])
    window_dimensions, window_strides, padding = _inputs_Tindices
    _inputs_flat = [operand, window_dimensions, window_strides, padding, source, init_value]
    _attrs = ('T', _attr_T, 'Tindices', _attr_Tindices, 'select', select, 'scatter', scatter)
    _result = _execute.execute(b'XlaSelectAndScatter', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('XlaSelectAndScatter', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result