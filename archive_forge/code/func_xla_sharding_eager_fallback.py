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
def xla_sharding_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_XlaSharding_T], sharding: str, unspecified_dims, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_XlaSharding_T]:
    if sharding is None:
        sharding = ''
    sharding = _execute.make_str(sharding, 'sharding')
    if unspecified_dims is None:
        unspecified_dims = []
    if not isinstance(unspecified_dims, (list, tuple)):
        raise TypeError("Expected list for 'unspecified_dims' argument to 'xla_sharding' Op, not %r." % unspecified_dims)
    unspecified_dims = [_execute.make_int(_i, 'unspecified_dims') for _i in unspecified_dims]
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
    _inputs_flat = [input]
    _attrs = ('T', _attr_T, 'sharding', sharding, 'unspecified_dims', unspecified_dims)
    _result = _execute.execute(b'XlaSharding', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('XlaSharding', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result