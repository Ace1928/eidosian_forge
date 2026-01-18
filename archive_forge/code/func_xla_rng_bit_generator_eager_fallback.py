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
def xla_rng_bit_generator_eager_fallback(algorithm: _atypes.TensorFuzzingAnnotation[_atypes.Int32], initial_state: _atypes.TensorFuzzingAnnotation[_atypes.UInt64], shape: _atypes.TensorFuzzingAnnotation[TV_XlaRngBitGenerator_Tshape], dtype: TV_XlaRngBitGenerator_dtype, name, ctx):
    if dtype is None:
        dtype = _dtypes.uint64
    dtype = _execute.make_type(dtype, 'dtype')
    _attr_Tshape, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    algorithm = _ops.convert_to_tensor(algorithm, _dtypes.int32)
    initial_state = _ops.convert_to_tensor(initial_state, _dtypes.uint64)
    _inputs_flat = [algorithm, initial_state, shape]
    _attrs = ('dtype', dtype, 'Tshape', _attr_Tshape)
    _result = _execute.execute(b'XlaRngBitGenerator', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('XlaRngBitGenerator', _inputs_flat, _attrs, _result)
    _result = _XlaRngBitGeneratorOutput._make(_result)
    return _result