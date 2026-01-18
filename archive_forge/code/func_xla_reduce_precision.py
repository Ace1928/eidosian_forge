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
@tf_export('xla_reduce_precision')
def xla_reduce_precision(operand: _atypes.TensorFuzzingAnnotation[TV_XlaReducePrecision_T], exponent_bits: int, mantissa_bits: int, name=None) -> _atypes.TensorFuzzingAnnotation[TV_XlaReducePrecision_T]:
    """Wraps the XLA ReducePrecision operator

    documented at https://www.tensorflow.org/xla/operation_semantics#reduceprecision.

  Args:
    operand: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
      array of floating-point type.
    exponent_bits: An `int`. number of exponent bits in lower-precision format
    mantissa_bits: An `int`. number of mantissa bits in lower-precision format
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `operand`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaReducePrecision', name, operand, 'exponent_bits', exponent_bits, 'mantissa_bits', mantissa_bits)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_reduce_precision((operand, exponent_bits, mantissa_bits, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_reduce_precision_eager_fallback(operand, exponent_bits=exponent_bits, mantissa_bits=mantissa_bits, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_reduce_precision, (), dict(operand=operand, exponent_bits=exponent_bits, mantissa_bits=mantissa_bits, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_reduce_precision((operand, exponent_bits, mantissa_bits, name), None)
        if _result is not NotImplemented:
            return _result
    exponent_bits = _execute.make_int(exponent_bits, 'exponent_bits')
    mantissa_bits = _execute.make_int(mantissa_bits, 'mantissa_bits')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaReducePrecision', operand=operand, exponent_bits=exponent_bits, mantissa_bits=mantissa_bits, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_reduce_precision, (), dict(operand=operand, exponent_bits=exponent_bits, mantissa_bits=mantissa_bits, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'exponent_bits', _op._get_attr_int('exponent_bits'), 'mantissa_bits', _op._get_attr_int('mantissa_bits'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaReducePrecision', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result