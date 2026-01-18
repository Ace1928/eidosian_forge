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
def xla_variadic_sort_eager_fallback(inputs, dimension: _atypes.TensorFuzzingAnnotation[_atypes.Int32], comparator, is_stable: bool, name, ctx):
    is_stable = _execute.make_bool(is_stable, 'is_stable')
    _attr_T, inputs = _execute.convert_to_mixed_eager_tensors(inputs, ctx)
    dimension = _ops.convert_to_tensor(dimension, _dtypes.int32)
    _inputs_flat = list(inputs) + [dimension]
    _attrs = ('T', _attr_T, 'comparator', comparator, 'is_stable', is_stable)
    _result = _execute.execute(b'XlaVariadicSort', len(inputs), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('XlaVariadicSort', _inputs_flat, _attrs, _result)
    return _result