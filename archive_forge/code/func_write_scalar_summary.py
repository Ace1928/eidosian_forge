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
def write_scalar_summary(writer: _atypes.TensorFuzzingAnnotation[_atypes.Resource], step: _atypes.TensorFuzzingAnnotation[_atypes.Int64], tag: _atypes.TensorFuzzingAnnotation[_atypes.String], value: _atypes.TensorFuzzingAnnotation[TV_WriteScalarSummary_T], name=None):
    """Writes a scalar summary.

  Writes scalar `value` at `step` with `tag` using summary `writer`.

  Args:
    writer: A `Tensor` of type `resource`.
    step: A `Tensor` of type `int64`.
    tag: A `Tensor` of type `string`.
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'WriteScalarSummary', name, writer, step, tag, value)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return write_scalar_summary_eager_fallback(writer, step, tag, value, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('WriteScalarSummary', writer=writer, step=step, tag=tag, value=value, name=name)
    return _op