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
@tf_export('xla_send')
def xla_send(tensor: _atypes.TensorFuzzingAnnotation[TV_XlaSend_T], tensor_name: str, name=None):
    """Sends the named tensor to another XLA computation. Wraps the XLA Send operator

  documented at
   https://www.tensorflow.org/performance/xla/operation_semantics#send .

  Args:
    tensor: A `Tensor`. The tensor to send.
    tensor_name: A `string`. A string key that identifies the channel.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaSend', name, tensor, 'tensor_name', tensor_name)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_send((tensor, tensor_name, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_send_eager_fallback(tensor, tensor_name=tensor_name, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_send, (), dict(tensor=tensor, tensor_name=tensor_name, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_send((tensor, tensor_name, name), None)
        if _result is not NotImplemented:
            return _result
    tensor_name = _execute.make_str(tensor_name, 'tensor_name')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaSend', tensor=tensor, tensor_name=tensor_name, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_send, (), dict(tensor=tensor, tensor_name=tensor_name, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    return _op