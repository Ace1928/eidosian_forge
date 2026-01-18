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
def to_bool(input: _atypes.TensorFuzzingAnnotation[TV_ToBool_T], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Bool]:
    """Converts a tensor to a scalar predicate.

  Converts a tensor to a scalar predicate with the following rules:

  - For 0D tensors, truthiness is determined by comparing against a "zero"
    value. For numerical types it is the obvious zero. For strings it is the
    empty string.

  - For >0D tensors, truthiness is determined by looking at the number of
    elements. If has zero elements, then the result is false. Otherwise the
    result is true.

  This matches the behavior of If and While for determining if a tensor counts
  as true/false for a branch condition.

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ToBool', name, input)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return to_bool_eager_fallback(input, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ToBool', input=input, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ToBool', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result