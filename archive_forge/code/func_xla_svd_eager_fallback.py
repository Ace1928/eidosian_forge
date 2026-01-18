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
def xla_svd_eager_fallback(a: _atypes.TensorFuzzingAnnotation[TV_XlaSvd_T], max_iter: int, epsilon: float, precision_config: str, name, ctx):
    max_iter = _execute.make_int(max_iter, 'max_iter')
    epsilon = _execute.make_float(epsilon, 'epsilon')
    precision_config = _execute.make_str(precision_config, 'precision_config')
    _attr_T, (a,) = _execute.args_to_matching_eager([a], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64])
    _inputs_flat = [a]
    _attrs = ('max_iter', max_iter, 'epsilon', epsilon, 'precision_config', precision_config, 'T', _attr_T)
    _result = _execute.execute(b'XlaSvd', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('XlaSvd', _inputs_flat, _attrs, _result)
    _result = _XlaSvdOutput._make(_result)
    return _result