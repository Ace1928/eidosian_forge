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
def tpu_partitioned_input_v2_eager_fallback(inputs: List[_atypes.TensorFuzzingAnnotation[TV_TPUPartitionedInputV2_T]], partition_dims, is_packed: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_TPUPartitionedInputV2_T]:
    if not isinstance(inputs, (list, tuple)):
        raise TypeError("Expected list for 'inputs' argument to 'tpu_partitioned_input_v2' Op, not %r." % inputs)
    _attr_N = len(inputs)
    if not isinstance(partition_dims, (list, tuple)):
        raise TypeError("Expected list for 'partition_dims' argument to 'tpu_partitioned_input_v2' Op, not %r." % partition_dims)
    partition_dims = [_execute.make_int(_i, 'partition_dims') for _i in partition_dims]
    if is_packed is None:
        is_packed = False
    is_packed = _execute.make_bool(is_packed, 'is_packed')
    _attr_T, inputs = _execute.args_to_matching_eager(list(inputs), ctx, [])
    _inputs_flat = list(inputs)
    _attrs = ('N', _attr_N, 'T', _attr_T, 'partition_dims', partition_dims, 'is_packed', is_packed)
    _result = _execute.execute(b'TPUPartitionedInputV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TPUPartitionedInputV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result