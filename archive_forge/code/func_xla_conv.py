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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_conv')
def xla_conv(lhs: _atypes.TensorFuzzingAnnotation[TV_XlaConv_T], rhs: _atypes.TensorFuzzingAnnotation[TV_XlaConv_T], window_strides: _atypes.TensorFuzzingAnnotation[TV_XlaConv_Tindices], padding: _atypes.TensorFuzzingAnnotation[TV_XlaConv_Tindices], lhs_dilation: _atypes.TensorFuzzingAnnotation[TV_XlaConv_Tindices], rhs_dilation: _atypes.TensorFuzzingAnnotation[TV_XlaConv_Tindices], feature_group_count: _atypes.TensorFuzzingAnnotation[TV_XlaConv_Tindices], dimension_numbers: str, precision_config: str, name=None) -> _atypes.TensorFuzzingAnnotation[TV_XlaConv_T]:
    """Wraps the XLA ConvGeneralDilated operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#conv_convolution
  .

  Args:
    lhs: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the input tensor
    rhs: A `Tensor`. Must have the same type as `lhs`. the kernel tensor
    window_strides: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      the inter-window strides
    padding: A `Tensor`. Must have the same type as `window_strides`.
      the padding to apply at the start and end of each input dimensions
    lhs_dilation: A `Tensor`. Must have the same type as `window_strides`.
      dilation to apply between input elements
    rhs_dilation: A `Tensor`. Must have the same type as `window_strides`.
      dilation to apply between kernel elements
    feature_group_count: A `Tensor`. Must have the same type as `window_strides`.
      number of feature groups for grouped convolution.
    dimension_numbers: A `string`.
      a serialized xla::ConvolutionDimensionNumbers proto.
    precision_config: A `string`. a serialized xla::PrecisionConfig proto.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `lhs`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaConv', name, lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count, 'dimension_numbers', dimension_numbers, 'precision_config', precision_config)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_conv((lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count, dimension_numbers, precision_config, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_conv_eager_fallback(lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count, dimension_numbers=dimension_numbers, precision_config=precision_config, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_conv, (), dict(lhs=lhs, rhs=rhs, window_strides=window_strides, padding=padding, lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation, feature_group_count=feature_group_count, dimension_numbers=dimension_numbers, precision_config=precision_config, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_conv((lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count, dimension_numbers, precision_config, name), None)
        if _result is not NotImplemented:
            return _result
    dimension_numbers = _execute.make_str(dimension_numbers, 'dimension_numbers')
    precision_config = _execute.make_str(precision_config, 'precision_config')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaConv', lhs=lhs, rhs=rhs, window_strides=window_strides, padding=padding, lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation, feature_group_count=feature_group_count, dimension_numbers=dimension_numbers, precision_config=precision_config, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_conv, (), dict(lhs=lhs, rhs=rhs, window_strides=window_strides, padding=padding, lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation, feature_group_count=feature_group_count, dimension_numbers=dimension_numbers, precision_config=precision_config, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tindices', _op._get_attr_type('Tindices'), 'dimension_numbers', _op.get_attr('dimension_numbers'), 'precision_config', _op.get_attr('precision_config'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaConv', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result