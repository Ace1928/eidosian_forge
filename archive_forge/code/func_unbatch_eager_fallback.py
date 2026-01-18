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
def unbatch_eager_fallback(batched_tensor: _atypes.TensorFuzzingAnnotation[TV_Unbatch_T], batch_index: _atypes.TensorFuzzingAnnotation[_atypes.Int64], id: _atypes.TensorFuzzingAnnotation[_atypes.Int64], timeout_micros: int, container: str, shared_name: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_Unbatch_T]:
    timeout_micros = _execute.make_int(timeout_micros, 'timeout_micros')
    if container is None:
        container = ''
    container = _execute.make_str(container, 'container')
    if shared_name is None:
        shared_name = ''
    shared_name = _execute.make_str(shared_name, 'shared_name')
    _attr_T, (batched_tensor,) = _execute.args_to_matching_eager([batched_tensor], ctx, [])
    batch_index = _ops.convert_to_tensor(batch_index, _dtypes.int64)
    id = _ops.convert_to_tensor(id, _dtypes.int64)
    _inputs_flat = [batched_tensor, batch_index, id]
    _attrs = ('timeout_micros', timeout_micros, 'container', container, 'shared_name', shared_name, 'T', _attr_T)
    _result = _execute.execute(b'Unbatch', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('Unbatch', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result