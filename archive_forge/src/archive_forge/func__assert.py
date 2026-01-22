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
def _assert(condition: _atypes.TensorFuzzingAnnotation[_atypes.Bool], data, summarize: int=3, name=None):
    """Asserts that the given condition is true.

  If `condition` evaluates to false, print the list of tensors in `data`.
  `summarize` determines how many entries of the tensors to print.

  Args:
    condition: A `Tensor` of type `bool`. The condition to evaluate.
    data: A list of `Tensor` objects.
      The tensors to print out when condition is false.
    summarize: An optional `int`. Defaults to `3`.
      Print this many entries of each tensor.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Assert', name, condition, data, 'summarize', summarize)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return _assert_eager_fallback(condition, data, summarize=summarize, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if summarize is None:
        summarize = 3
    summarize = _execute.make_int(summarize, 'summarize')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('Assert', condition=condition, data=data, summarize=summarize, name=name)
    return _op