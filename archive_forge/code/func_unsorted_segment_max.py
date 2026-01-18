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
@tf_export('math.unsorted_segment_max', v1=['math.unsorted_segment_max', 'unsorted_segment_max'])
@deprecated_endpoints('unsorted_segment_max')
def unsorted_segment_max(data: _atypes.TensorFuzzingAnnotation[TV_UnsortedSegmentMax_T], segment_ids: _atypes.TensorFuzzingAnnotation[TV_UnsortedSegmentMax_Tindices], num_segments: _atypes.TensorFuzzingAnnotation[TV_UnsortedSegmentMax_Tnumsegments], name=None) -> _atypes.TensorFuzzingAnnotation[TV_UnsortedSegmentMax_T]:
    """Computes the maximum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  This operator is similar to `tf.math.unsorted_segment_sum`,
  Instead of computing the sum over segments, it computes the maximum such that:

  \\\\(output_i = \\max_{j...} data[j...]\\\\) where max is over tuples `j...` such
  that `segment_ids[j...] == i`.

  If the maximum is empty for a given segment ID `i`, it outputs the smallest
  possible value for the specific numeric type,
  `output[i] = numeric_limits<T>::lowest()`.

  If the given segment ID `i` is negative, then the corresponding value is
  dropped, and will not be included in the result.

  Caution: On CPU, values in `segment_ids` are always validated to be less than
  `num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
  does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
  result in safe but unspecified behavior, which may include ignoring
  out-of-bound indices or outputting a tensor with a 0 stored in the first
  dimension of its shape if `num_segments` is 0.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentMax.png" alt>
  </div>

  For example:

  >>> c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
  >>> tf.math.unsorted_segment_max(c, tf.constant([0, 1, 0]), num_segments=2).numpy()
  array([[4, 3, 3, 4],
         [5,  6, 7, 8]], dtype=int32)

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A tensor whose shape is a prefix of `data.shape`.
      The values must be less than `num_segments`.

      Caution: The values are always validated to be in range on CPU, never validated
      on GPU.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'UnsortedSegmentMax', name, data, segment_ids, num_segments)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_unsorted_segment_max((data, segment_ids, num_segments, name), None)
            if _result is not NotImplemented:
                return _result
            return unsorted_segment_max_eager_fallback(data, segment_ids, num_segments, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(unsorted_segment_max, (), dict(data=data, segment_ids=segment_ids, num_segments=num_segments, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_unsorted_segment_max((data, segment_ids, num_segments, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('UnsortedSegmentMax', data=data, segment_ids=segment_ids, num_segments=num_segments, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(unsorted_segment_max, (), dict(data=data, segment_ids=segment_ids, num_segments=num_segments, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tindices', _op._get_attr_type('Tindices'), 'Tnumsegments', _op._get_attr_type('Tnumsegments'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('UnsortedSegmentMax', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result