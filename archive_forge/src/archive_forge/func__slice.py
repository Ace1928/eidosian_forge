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
def _slice(input: _atypes.TensorFuzzingAnnotation[TV_Slice_T], begin: _atypes.TensorFuzzingAnnotation[TV_Slice_Index], size: _atypes.TensorFuzzingAnnotation[TV_Slice_Index], name=None) -> _atypes.TensorFuzzingAnnotation[TV_Slice_T]:
    """Return a slice from 'input'.

  The output tensor is a tensor with dimensions described by 'size'
  whose values are extracted from 'input' starting at the offsets in
  'begin'.

  *Requirements*:
    0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)

  Args:
    input: A `Tensor`.
    begin: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      begin[i] specifies the offset into the 'i'th dimension of
      'input' to slice from.
    size: A `Tensor`. Must have the same type as `begin`.
      size[i] specifies the number of elements of the 'i'th dimension
      of 'input' to slice. If size[i] is -1, all remaining elements in dimension
      i are included in the slice (i.e. this is equivalent to setting
      size[i] = input.dim_size(i) - begin[i]).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Slice', name, input, begin, size)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return _slice_eager_fallback(input, begin, size, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('Slice', input=input, begin=begin, size=size, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Index', _op._get_attr_type('Index'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('Slice', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result