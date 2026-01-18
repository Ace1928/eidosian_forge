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
def upper_bound(sorted_inputs: _atypes.TensorFuzzingAnnotation[TV_UpperBound_T], values: _atypes.TensorFuzzingAnnotation[TV_UpperBound_T], out_type: TV_UpperBound_out_type=_dtypes.int32, name=None) -> _atypes.TensorFuzzingAnnotation[TV_UpperBound_out_type]:
    """Applies upper_bound(sorted_search_values, values) along each row.

  Each set of rows with the same index in (sorted_inputs, values) is treated
  independently.  The resulting row is the equivalent of calling
  `np.searchsorted(sorted_inputs, values, side='right')`.

  The result is not a global index to the entire
  `Tensor`, but rather just the index in the last dimension.

  A 2-D example:
    sorted_sequence = [[0, 3, 9, 9, 10],
                       [1, 2, 3, 4, 5]]
    values = [[2, 4, 9],
              [0, 2, 6]]

    result = UpperBound(sorted_sequence, values)

    result == [[1, 2, 4],
               [0, 2, 5]]

  Args:
    sorted_inputs: A `Tensor`. 2-D Tensor where each row is ordered.
    values: A `Tensor`. Must have the same type as `sorted_inputs`.
      2-D Tensor with the same numbers of rows as `sorted_search_values`. Contains
      the values that will be searched for in `sorted_search_values`.
    out_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'UpperBound', name, sorted_inputs, values, 'out_type', out_type)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return upper_bound_eager_fallback(sorted_inputs, values, out_type=out_type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if out_type is None:
        out_type = _dtypes.int32
    out_type = _execute.make_type(out_type, 'out_type')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('UpperBound', sorted_inputs=sorted_inputs, values=values, out_type=out_type, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'out_type', _op._get_attr_type('out_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('UpperBound', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result