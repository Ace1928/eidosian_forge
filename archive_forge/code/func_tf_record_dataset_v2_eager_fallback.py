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
def tf_record_dataset_v2_eager_fallback(filenames: _atypes.TensorFuzzingAnnotation[_atypes.String], compression_type: _atypes.TensorFuzzingAnnotation[_atypes.String], buffer_size: _atypes.TensorFuzzingAnnotation[_atypes.Int64], byte_offsets: _atypes.TensorFuzzingAnnotation[_atypes.Int64], metadata: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    filenames = _ops.convert_to_tensor(filenames, _dtypes.string)
    compression_type = _ops.convert_to_tensor(compression_type, _dtypes.string)
    buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
    byte_offsets = _ops.convert_to_tensor(byte_offsets, _dtypes.int64)
    _inputs_flat = [filenames, compression_type, buffer_size, byte_offsets]
    _attrs = ('metadata', metadata)
    _result = _execute.execute(b'TFRecordDatasetV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TFRecordDatasetV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result