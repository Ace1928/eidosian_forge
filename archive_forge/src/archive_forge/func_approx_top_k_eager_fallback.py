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
def approx_top_k_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_ApproxTopK_T], k: int, reduction_dimension: int, recall_target: float, is_max_k: bool, reduction_input_size_override: int, aggregate_to_topk: bool, name, ctx):
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
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32])
    _inputs_flat = [input]
    _attrs = ('k', k, 'reduction_dimension', reduction_dimension, 'recall_target', recall_target, 'is_max_k', is_max_k, 'reduction_input_size_override', reduction_input_size_override, 'aggregate_to_topk', aggregate_to_topk, 'T', _attr_T)
    _result = _execute.execute(b'ApproxTopK', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ApproxTopK', _inputs_flat, _attrs, _result)
    _result = _ApproxTopKOutput._make(_result)
    return _result