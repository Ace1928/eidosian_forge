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
def tpu_replicated_input_eager_fallback(inputs: List[_atypes.TensorFuzzingAnnotation[TV_TPUReplicatedInput_T]], is_mirrored_variable: bool, index: int, is_packed: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_TPUReplicatedInput_T]:
    if not isinstance(inputs, (list, tuple)):
        raise TypeError("Expected list for 'inputs' argument to 'tpu_replicated_input' Op, not %r." % inputs)
    _attr_N = len(inputs)
    if is_mirrored_variable is None:
        is_mirrored_variable = False
    is_mirrored_variable = _execute.make_bool(is_mirrored_variable, 'is_mirrored_variable')
    if index is None:
        index = -1
    index = _execute.make_int(index, 'index')
    if is_packed is None:
        is_packed = False
    is_packed = _execute.make_bool(is_packed, 'is_packed')
    _attr_T, inputs = _execute.args_to_matching_eager(list(inputs), ctx, [])
    _inputs_flat = list(inputs)
    _attrs = ('N', _attr_N, 'T', _attr_T, 'is_mirrored_variable', is_mirrored_variable, 'index', index, 'is_packed', is_packed)
    _result = _execute.execute(b'TPUReplicatedInput', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TPUReplicatedInput', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result