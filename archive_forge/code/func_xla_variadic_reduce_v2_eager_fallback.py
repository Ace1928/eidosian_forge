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
def xla_variadic_reduce_v2_eager_fallback(inputs, init_values, dimensions_to_reduce, reducer, name, ctx):
    if not isinstance(dimensions_to_reduce, (list, tuple)):
        raise TypeError("Expected list for 'dimensions_to_reduce' argument to 'xla_variadic_reduce_v2' Op, not %r." % dimensions_to_reduce)
    dimensions_to_reduce = [_execute.make_int(_i, 'dimensions_to_reduce') for _i in dimensions_to_reduce]
    _attr_T, (inputs, init_values) = _execute.args_to_mixed_eager_tensors((inputs, init_values), ctx)
    _inputs_flat = list(inputs) + list(init_values)
    _attrs = ('T', _attr_T, 'dimensions_to_reduce', dimensions_to_reduce, 'reducer', reducer)
    _result = _execute.execute(b'XlaVariadicReduceV2', len(inputs), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('XlaVariadicReduceV2', _inputs_flat, _attrs, _result)
    return _result