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
def top_kv2(input: _atypes.TensorFuzzingAnnotation[TV_TopKV2_T], k: _atypes.TensorFuzzingAnnotation[TV_TopKV2_Tk], sorted: bool=True, index_type: TV_TopKV2_index_type=_dtypes.int32, name=None):
    """Finds values and indices of the `k` largest elements for the last dimension.

  If the input is a vector (rank-1), finds the `k` largest entries in the vector
  and outputs their values and indices as vectors.  Thus `values[j]` is the
  `j`-th largest entry in `input`, and its index is `indices[j]`.

  For matrices (resp. higher rank input), computes the top `k` entries in each
  row (resp. vector along the last dimension).  Thus,

      values.shape = indices.shape = input.shape[:-1] + [k]

  If two elements are equal, the lower-index element appears first.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      1-D or higher with last dimension at least `k`.
    k: A `Tensor`. Must be one of the following types: `int16`, `int32`, `int64`.
      0-D.  Number of top elements to look for along the last dimension (along each
      row for matrices).
    sorted: An optional `bool`. Defaults to `True`.
      If true the resulting `k` elements will be sorted by the values in
      descending order.
    index_type: An optional `tf.DType` from: `tf.int16, tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (values, indices).

    values: A `Tensor`. Has the same type as `input`.
    indices: A `Tensor` of type `index_type`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TopKV2', name, input, k, 'sorted', sorted, 'index_type', index_type)
            _result = _TopKV2Output._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return top_kv2_eager_fallback(input, k, sorted=sorted, index_type=index_type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if sorted is None:
        sorted = True
    sorted = _execute.make_bool(sorted, 'sorted')
    if index_type is None:
        index_type = _dtypes.int32
    index_type = _execute.make_type(index_type, 'index_type')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TopKV2', input=input, k=k, sorted=sorted, index_type=index_type, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('sorted', _op._get_attr_bool('sorted'), 'T', _op._get_attr_type('T'), 'Tk', _op._get_attr_type('Tk'), 'index_type', _op._get_attr_type('index_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TopKV2', _inputs_flat, _attrs, _result)
    _result = _TopKV2Output._make(_result)
    return _result