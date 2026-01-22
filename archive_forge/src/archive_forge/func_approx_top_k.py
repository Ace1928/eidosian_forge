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
@tf_export('approx_top_k')
def approx_top_k(input: _atypes.TensorFuzzingAnnotation[TV_ApproxTopK_T], k: int, reduction_dimension: int=-1, recall_target: float=0.95, is_max_k: bool=True, reduction_input_size_override: int=-1, aggregate_to_topk: bool=True, name=None):
    """Returns min/max k values and their indices of the input operand in an approximate manner.

  See https://arxiv.org/abs/2206.14286 for the algorithm details.
  This op is only optimized on TPU currently.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      Array to search. Must be at least 1-D of the floating type
    k: An `int` that is `>= 0`. Specifies the number of min/max-k.
    reduction_dimension: An optional `int`. Defaults to `-1`.
      Integer dimension along which to search. Default: -1.
    recall_target: An optional `float`. Defaults to `0.95`.
      Recall target for the approximation. Range in (0,1]
    is_max_k: An optional `bool`. Defaults to `True`.
      When true, computes max-k; otherwise computes min-k.
    reduction_input_size_override: An optional `int`. Defaults to `-1`.
      When set to a positive value, it overrides the size determined by
      `input[reduction_dim]` for evaluating the recall. This option is useful when
      the given `input` is only a subset of the overall computation in SPMD or
      distributed pipelines, where the true input size cannot be deferred by the
      `input` shape.
    aggregate_to_topk: An optional `bool`. Defaults to `True`.
      When true, aggregates approximate results to top-k. When false, returns the
      approximate results. The number of the approximate results is implementation
      defined and is greater equals to the specified `k`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (values, indices).

    values: A `Tensor`. Has the same type as `input`.
    indices: A `Tensor` of type `int32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ApproxTopK', name, input, 'k', k, 'reduction_dimension', reduction_dimension, 'recall_target', recall_target, 'is_max_k', is_max_k, 'reduction_input_size_override', reduction_input_size_override, 'aggregate_to_topk', aggregate_to_topk)
            _result = _ApproxTopKOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_approx_top_k((input, k, reduction_dimension, recall_target, is_max_k, reduction_input_size_override, aggregate_to_topk, name), None)
            if _result is not NotImplemented:
                return _result
            return approx_top_k_eager_fallback(input, k=k, reduction_dimension=reduction_dimension, recall_target=recall_target, is_max_k=is_max_k, reduction_input_size_override=reduction_input_size_override, aggregate_to_topk=aggregate_to_topk, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(approx_top_k, (), dict(input=input, k=k, reduction_dimension=reduction_dimension, recall_target=recall_target, is_max_k=is_max_k, reduction_input_size_override=reduction_input_size_override, aggregate_to_topk=aggregate_to_topk, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_approx_top_k((input, k, reduction_dimension, recall_target, is_max_k, reduction_input_size_override, aggregate_to_topk, name), None)
        if _result is not NotImplemented:
            return _result
    k = _execute.make_int(k, 'k')
    if reduction_dimension is None:
        reduction_dimension = -1
    reduction_dimension = _execute.make_int(reduction_dimension, 'reduction_dimension')
    if recall_target is None:
        recall_target = 0.95
    recall_target = _execute.make_float(recall_target, 'recall_target')
    if is_max_k is None:
        is_max_k = True
    is_max_k = _execute.make_bool(is_max_k, 'is_max_k')
    if reduction_input_size_override is None:
        reduction_input_size_override = -1
    reduction_input_size_override = _execute.make_int(reduction_input_size_override, 'reduction_input_size_override')
    if aggregate_to_topk is None:
        aggregate_to_topk = True
    aggregate_to_topk = _execute.make_bool(aggregate_to_topk, 'aggregate_to_topk')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('ApproxTopK', input=input, k=k, reduction_dimension=reduction_dimension, recall_target=recall_target, is_max_k=is_max_k, reduction_input_size_override=reduction_input_size_override, aggregate_to_topk=aggregate_to_topk, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(approx_top_k, (), dict(input=input, k=k, reduction_dimension=reduction_dimension, recall_target=recall_target, is_max_k=is_max_k, reduction_input_size_override=reduction_input_size_override, aggregate_to_topk=aggregate_to_topk, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('k', _op._get_attr_int('k'), 'reduction_dimension', _op._get_attr_int('reduction_dimension'), 'recall_target', _op.get_attr('recall_target'), 'is_max_k', _op._get_attr_bool('is_max_k'), 'reduction_input_size_override', _op._get_attr_int('reduction_input_size_override'), 'aggregate_to_topk', _op._get_attr_bool('aggregate_to_topk'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ApproxTopK', _inputs_flat, _attrs, _result)
    _result = _ApproxTopKOutput._make(_result)
    return _result