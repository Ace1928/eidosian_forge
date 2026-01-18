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
def xla_key_value_sort_eager_fallback(keys: _atypes.TensorFuzzingAnnotation[TV_XlaKeyValueSort_K], values: _atypes.TensorFuzzingAnnotation[TV_XlaKeyValueSort_V], name, ctx):
    _attr_K, (keys,) = _execute.args_to_matching_eager([keys], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64])
    _attr_V, (values,) = _execute.args_to_matching_eager([values], ctx, [])
    _inputs_flat = [keys, values]
    _attrs = ('K', _attr_K, 'V', _attr_V)
    _result = _execute.execute(b'XlaKeyValueSort', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('XlaKeyValueSort', _inputs_flat, _attrs, _result)
    _result = _XlaKeyValueSortOutput._make(_result)
    return _result