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
@tf_export('xla_einsum')
def xla_einsum(a: _atypes.TensorFuzzingAnnotation[TV_XlaEinsum_T], b: _atypes.TensorFuzzingAnnotation[TV_XlaEinsum_T], equation: str, name=None) -> _atypes.TensorFuzzingAnnotation[TV_XlaEinsum_T]:
    """An op which supports basic einsum op with 2 inputs and 1 output.

  This op has better TPU performance since it doesn't have explicitly reshape and
  transpose operations as tf.einsum does.

  Args:
    a: A `Tensor`. Must be one of the following types: `complex64`, `bfloat16`, `float32`.
    b: A `Tensor`. Must have the same type as `a`.
    equation: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaEinsum', name, a, b, 'equation', equation)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_einsum((a, b, equation, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_einsum_eager_fallback(a, b, equation=equation, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_einsum, (), dict(a=a, b=b, equation=equation, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_einsum((a, b, equation, name), None)
        if _result is not NotImplemented:
            return _result
    equation = _execute.make_str(equation, 'equation')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaEinsum', a=a, b=b, equation=equation, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_einsum, (), dict(a=a, b=b, equation=equation, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('equation', _op.get_attr('equation'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaEinsum', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result