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
@tf_export('xla_reduce_scatter')
def xla_reduce_scatter(input: _atypes.TensorFuzzingAnnotation[TV_XlaReduceScatter_T], group_assignment: _atypes.TensorFuzzingAnnotation[_atypes.Int32], scatter_dimension: _atypes.TensorFuzzingAnnotation[_atypes.Int32], reduce_op: str, name=None) -> _atypes.TensorFuzzingAnnotation[TV_XlaReduceScatter_T]:
    """Wraps the XLA ReduceScatter operator

    documented at https://www.tensorflow.org/xla/operation_semantics#reducescatter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `int32`, `uint32`.
      Array or a non-empty tuple of arrays to reduce across replicas.
    group_assignment: A `Tensor` of type `int32`.
      Groups between which the reductions are performed.
    scatter_dimension: A `Tensor` of type `int32`. Dimension to scatter.
    reduce_op: A `string` from: `"Min", "Max", "Mul", "Add", "Mean"`.
      Reduction computation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaReduceScatter', name, input, group_assignment, scatter_dimension, 'reduce_op', reduce_op)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_reduce_scatter((input, group_assignment, scatter_dimension, reduce_op, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_reduce_scatter_eager_fallback(input, group_assignment, scatter_dimension, reduce_op=reduce_op, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_reduce_scatter, (), dict(input=input, group_assignment=group_assignment, scatter_dimension=scatter_dimension, reduce_op=reduce_op, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_reduce_scatter((input, group_assignment, scatter_dimension, reduce_op, name), None)
        if _result is not NotImplemented:
            return _result
    reduce_op = _execute.make_str(reduce_op, 'reduce_op')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaReduceScatter', input=input, group_assignment=group_assignment, scatter_dimension=scatter_dimension, reduce_op=reduce_op, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_reduce_scatter, (), dict(input=input, group_assignment=group_assignment, scatter_dimension=scatter_dimension, reduce_op=reduce_op, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'reduce_op', _op.get_attr('reduce_op'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaReduceScatter', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result