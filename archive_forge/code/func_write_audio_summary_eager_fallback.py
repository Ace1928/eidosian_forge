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
def write_audio_summary_eager_fallback(writer: _atypes.TensorFuzzingAnnotation[_atypes.Resource], step: _atypes.TensorFuzzingAnnotation[_atypes.Int64], tag: _atypes.TensorFuzzingAnnotation[_atypes.String], tensor: _atypes.TensorFuzzingAnnotation[_atypes.Float32], sample_rate: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_outputs: int, name, ctx):
    if max_outputs is None:
        max_outputs = 3
    max_outputs = _execute.make_int(max_outputs, 'max_outputs')
    writer = _ops.convert_to_tensor(writer, _dtypes.resource)
    step = _ops.convert_to_tensor(step, _dtypes.int64)
    tag = _ops.convert_to_tensor(tag, _dtypes.string)
    tensor = _ops.convert_to_tensor(tensor, _dtypes.float32)
    sample_rate = _ops.convert_to_tensor(sample_rate, _dtypes.float32)
    _inputs_flat = [writer, step, tag, tensor, sample_rate]
    _attrs = ('max_outputs', max_outputs)
    _result = _execute.execute(b'WriteAudioSummary', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result