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
def tpu_partitioned_output_v2_eager_fallback(inputs: _atypes.TensorFuzzingAnnotation[TV_TPUPartitionedOutputV2_T], num_splits: int, partition_dims, name, ctx):
    num_splits = _execute.make_int(num_splits, 'num_splits')
    if not isinstance(partition_dims, (list, tuple)):
        raise TypeError("Expected list for 'partition_dims' argument to 'tpu_partitioned_output_v2' Op, not %r." % partition_dims)
    partition_dims = [_execute.make_int(_i, 'partition_dims') for _i in partition_dims]
    _attr_T, (inputs,) = _execute.args_to_matching_eager([inputs], ctx, [])
    _inputs_flat = [inputs]
    _attrs = ('T', _attr_T, 'num_splits', num_splits, 'partition_dims', partition_dims)
    _result = _execute.execute(b'TPUPartitionedOutputV2', num_splits, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TPUPartitionedOutputV2', _inputs_flat, _attrs, _result)
    return _result