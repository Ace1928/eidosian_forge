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
@tf_export('xla_set_bound')
def xla_set_bound(input: _atypes.TensorFuzzingAnnotation[_atypes.Int32], bound: _atypes.TensorFuzzingAnnotation[_atypes.Int32], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Int32]:
    """Set a bound for the given input value as a hint to Xla compiler,

          returns the same value.

  Args:
    input: A `Tensor` of type `int32`.
    bound: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaSetBound', name, input, bound)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_set_bound((input, bound, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_set_bound_eager_fallback(input, bound, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_set_bound, (), dict(input=input, bound=bound, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_set_bound((input, bound, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaSetBound', input=input, bound=bound, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_set_bound, (), dict(input=input, bound=bound, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ()
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaSetBound', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result