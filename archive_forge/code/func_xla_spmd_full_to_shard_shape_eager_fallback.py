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
def xla_spmd_full_to_shard_shape_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_XlaSpmdFullToShardShape_T], manual_sharding: str, dim: int, unspecified_dims, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_XlaSpmdFullToShardShape_T]:
    manual_sharding = _execute.make_str(manual_sharding, 'manual_sharding')
    if dim is None:
        dim = -1
    dim = _execute.make_int(dim, 'dim')
    if unspecified_dims is None:
        unspecified_dims = []
    if not isinstance(unspecified_dims, (list, tuple)):
        raise TypeError("Expected list for 'unspecified_dims' argument to 'xla_spmd_full_to_shard_shape' Op, not %r." % unspecified_dims)
    unspecified_dims = [_execute.make_int(_i, 'unspecified_dims') for _i in unspecified_dims]
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
    _inputs_flat = [input]
    _attrs = ('T', _attr_T, 'manual_sharding', manual_sharding, 'dim', dim, 'unspecified_dims', unspecified_dims)
    _result = _execute.execute(b'XlaSpmdFullToShardShape', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('XlaSpmdFullToShardShape', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result