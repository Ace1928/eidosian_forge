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
def abort(error_msg: str='', exit_without_error: bool=False, name=None):
    """Raise a exception to abort the process when called.

  If exit_without_error is true, the process will exit normally,
  otherwise it will exit with a SIGABORT signal.

  Returns nothing but an exception.

  Args:
    error_msg: An optional `string`. Defaults to `""`.
      A string which is the message associated with the exception.
    exit_without_error: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Abort', name, 'error_msg', error_msg, 'exit_without_error', exit_without_error)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return abort_eager_fallback(error_msg=error_msg, exit_without_error=exit_without_error, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if error_msg is None:
        error_msg = ''
    error_msg = _execute.make_str(error_msg, 'error_msg')
    if exit_without_error is None:
        exit_without_error = False
    exit_without_error = _execute.make_bool(exit_without_error, 'exit_without_error')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('Abort', error_msg=error_msg, exit_without_error=exit_without_error, name=name)
    return _op