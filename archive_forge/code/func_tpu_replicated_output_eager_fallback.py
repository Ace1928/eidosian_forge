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
def tpu_replicated_output_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_TPUReplicatedOutput_T], num_replicas: int, name, ctx):
    num_replicas = _execute.make_int(num_replicas, 'num_replicas')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
    _inputs_flat = [input]
    _attrs = ('num_replicas', num_replicas, 'T', _attr_T)
    _result = _execute.execute(b'TPUReplicatedOutput', num_replicas, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TPUReplicatedOutput', _inputs_flat, _attrs, _result)
    return _result