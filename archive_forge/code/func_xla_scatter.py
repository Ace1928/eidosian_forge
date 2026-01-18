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
@tf_export('xla_scatter')
def xla_scatter(operand: _atypes.TensorFuzzingAnnotation[TV_XlaScatter_T], scatter_indices: _atypes.TensorFuzzingAnnotation[TV_XlaScatter_Tindices], updates: _atypes.TensorFuzzingAnnotation[TV_XlaScatter_T], update_computation, dimension_numbers: str, indices_are_sorted: bool, name=None) -> _atypes.TensorFuzzingAnnotation[TV_XlaScatter_T]:
    """Wraps the XLA Scatter operator documented at

    https://www.tensorflow.org/xla/operation_semantics#scatter.

  Args:
    operand: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`, `bool`.
      Array to be scattered into.
    scatter_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Array containing the starting indices of the slices that must
      be scattered to.
    updates: A `Tensor`. Must have the same type as `operand`.
      Array containing the values that must be used for scattering.
    update_computation: A function decorated with @Defun.
      Computation to be used for combining the existing values in
      the input array and the updates during scatter.
    dimension_numbers: A `string`.
      A serialized xla::ScatterDimensionNumbers proto.
    indices_are_sorted: A `bool`.
      Boolean indicating if the indices are sorted.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `operand`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaScatter', name, operand, scatter_indices, updates, 'update_computation', update_computation, 'dimension_numbers', dimension_numbers, 'indices_are_sorted', indices_are_sorted)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_scatter((operand, scatter_indices, updates, update_computation, dimension_numbers, indices_are_sorted, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_scatter_eager_fallback(operand, scatter_indices, updates, update_computation=update_computation, dimension_numbers=dimension_numbers, indices_are_sorted=indices_are_sorted, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_scatter, (), dict(operand=operand, scatter_indices=scatter_indices, updates=updates, update_computation=update_computation, dimension_numbers=dimension_numbers, indices_are_sorted=indices_are_sorted, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_scatter((operand, scatter_indices, updates, update_computation, dimension_numbers, indices_are_sorted, name), None)
        if _result is not NotImplemented:
            return _result
    dimension_numbers = _execute.make_str(dimension_numbers, 'dimension_numbers')
    indices_are_sorted = _execute.make_bool(indices_are_sorted, 'indices_are_sorted')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaScatter', operand=operand, scatter_indices=scatter_indices, updates=updates, update_computation=update_computation, dimension_numbers=dimension_numbers, indices_are_sorted=indices_are_sorted, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_scatter, (), dict(operand=operand, scatter_indices=scatter_indices, updates=updates, update_computation=update_computation, dimension_numbers=dimension_numbers, indices_are_sorted=indices_are_sorted, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('update_computation', _op.get_attr('update_computation'), 'dimension_numbers', _op.get_attr('dimension_numbers'), 'indices_are_sorted', _op._get_attr_bool('indices_are_sorted'), 'T', _op._get_attr_type('T'), 'Tindices', _op._get_attr_type('Tindices'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaScatter', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result