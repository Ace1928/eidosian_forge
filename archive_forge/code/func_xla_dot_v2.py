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
@tf_export('xla_dot_v2')
def xla_dot_v2(lhs: _atypes.TensorFuzzingAnnotation[TV_XlaDotV2_LhsT], rhs: _atypes.TensorFuzzingAnnotation[TV_XlaDotV2_RhsT], dimension_numbers: str, precision_config: str, preferred_element_type: TV_XlaDotV2_preferred_element_type, name=None) -> _atypes.TensorFuzzingAnnotation[TV_XlaDotV2_preferred_element_type]:
    """Wraps the XLA DotGeneral operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#dotgeneral
  .

  Args:
    lhs: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the LHS tensor
    rhs: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the RHS tensor
    dimension_numbers: A `string`.
      a serialized xla::DotDimensionNumbers proto.
    precision_config: A `string`. a serialized xla::PrecisionConfig proto.
    preferred_element_type: A `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.complex64, tf.int64, tf.qint8, tf.quint8, tf.qint32, tf.bfloat16, tf.qint16, tf.quint16, tf.uint16, tf.complex128, tf.half, tf.uint32, tf.uint64`.
      The type of the tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `preferred_element_type`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaDotV2', name, lhs, rhs, 'dimension_numbers', dimension_numbers, 'precision_config', precision_config, 'preferred_element_type', preferred_element_type)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_dot_v2((lhs, rhs, dimension_numbers, precision_config, preferred_element_type, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_dot_v2_eager_fallback(lhs, rhs, dimension_numbers=dimension_numbers, precision_config=precision_config, preferred_element_type=preferred_element_type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_dot_v2, (), dict(lhs=lhs, rhs=rhs, dimension_numbers=dimension_numbers, precision_config=precision_config, preferred_element_type=preferred_element_type, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_dot_v2((lhs, rhs, dimension_numbers, precision_config, preferred_element_type, name), None)
        if _result is not NotImplemented:
            return _result
    dimension_numbers = _execute.make_str(dimension_numbers, 'dimension_numbers')
    precision_config = _execute.make_str(precision_config, 'precision_config')
    preferred_element_type = _execute.make_type(preferred_element_type, 'preferred_element_type')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaDotV2', lhs=lhs, rhs=rhs, dimension_numbers=dimension_numbers, precision_config=precision_config, preferred_element_type=preferred_element_type, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_dot_v2, (), dict(lhs=lhs, rhs=rhs, dimension_numbers=dimension_numbers, precision_config=precision_config, preferred_element_type=preferred_element_type, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('LhsT', _op._get_attr_type('LhsT'), 'RhsT', _op._get_attr_type('RhsT'), 'dimension_numbers', _op.get_attr('dimension_numbers'), 'precision_config', _op.get_attr('precision_config'), 'preferred_element_type', _op._get_attr_type('preferred_element_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaDotV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result