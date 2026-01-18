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
def write_raw_proto_summary_eager_fallback(writer: _atypes.TensorFuzzingAnnotation[_atypes.Resource], step: _atypes.TensorFuzzingAnnotation[_atypes.Int64], tensor: _atypes.TensorFuzzingAnnotation[_atypes.String], name, ctx):
    writer = _ops.convert_to_tensor(writer, _dtypes.resource)
    step = _ops.convert_to_tensor(step, _dtypes.int64)
    tensor = _ops.convert_to_tensor(tensor, _dtypes.string)
    _inputs_flat = [writer, step, tensor]
    _attrs = None
    _result = _execute.execute(b'WriteRawProtoSummary', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result