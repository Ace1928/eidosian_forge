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
@tf_export('xla_all_reduce')
def xla_all_reduce(input: _atypes.TensorFuzzingAnnotation[TV_XlaAllReduce_T], group_assignment: _atypes.TensorFuzzingAnnotation[_atypes.Int32], reduce_op: str, mode: str, name=None) -> _atypes.TensorFuzzingAnnotation[TV_XlaAllReduce_T]:
    """Wraps the XLA AllReduce operator

    documented at https://www.tensorflow.org/xla/operation_semantics#allreduce.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `int32`, `uint32`.
      Array or a non-empty tuple of arrays to reduce across replicas.
    group_assignment: A `Tensor` of type `int32`.
      Groups between which the reductions are performed.
    reduce_op: A `string` from: `"Min", "Max", "Mul", "Add", "Mean"`.
      Reduction computation.
    mode: A `string` from: `"CrossReplica", "CrossReplicaAndPartition"`.
      group mode.
      CrossReplica: group_assignment contains replica_id. Each group contains the
        replicas for the current partition.
      CrossReplicaAndPartition: group_assignment contains replica_id. Each group
        contains the replicas for all partitions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaAllReduce', name, input, group_assignment, 'reduce_op', reduce_op, 'mode', mode)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_all_reduce((input, group_assignment, reduce_op, mode, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_all_reduce_eager_fallback(input, group_assignment, reduce_op=reduce_op, mode=mode, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_all_reduce, (), dict(input=input, group_assignment=group_assignment, reduce_op=reduce_op, mode=mode, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_all_reduce((input, group_assignment, reduce_op, mode, name), None)
        if _result is not NotImplemented:
            return _result
    reduce_op = _execute.make_str(reduce_op, 'reduce_op')
    mode = _execute.make_str(mode, 'mode')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaAllReduce', input=input, group_assignment=group_assignment, reduce_op=reduce_op, mode=mode, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_all_reduce, (), dict(input=input, group_assignment=group_assignment, reduce_op=reduce_op, mode=mode, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'reduce_op', _op.get_attr('reduce_op'), 'mode', _op.get_attr('mode'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaAllReduce', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result