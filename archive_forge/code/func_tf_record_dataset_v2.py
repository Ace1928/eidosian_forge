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
def tf_record_dataset_v2(filenames: _atypes.TensorFuzzingAnnotation[_atypes.String], compression_type: _atypes.TensorFuzzingAnnotation[_atypes.String], buffer_size: _atypes.TensorFuzzingAnnotation[_atypes.Int64], byte_offsets: _atypes.TensorFuzzingAnnotation[_atypes.Int64], metadata: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Creates a dataset that emits the records from one or more TFRecord files.

  Args:
    filenames: A `Tensor` of type `string`.
      A scalar or vector containing the name(s) of the file(s) to be
      read.
    compression_type: A `Tensor` of type `string`.
      A scalar containing either (i) the empty string (no
      compression), (ii) "ZLIB", or (iii) "GZIP".
    buffer_size: A `Tensor` of type `int64`.
      A scalar representing the number of bytes to buffer. A value of
      0 means no buffering will be performed.
    byte_offsets: A `Tensor` of type `int64`.
      A scalar or vector containing the number of bytes for each file
      that will be skipped prior to reading.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TFRecordDatasetV2', name, filenames, compression_type, buffer_size, byte_offsets, 'metadata', metadata)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tf_record_dataset_v2_eager_fallback(filenames, compression_type, buffer_size, byte_offsets, metadata=metadata, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TFRecordDatasetV2', filenames=filenames, compression_type=compression_type, buffer_size=buffer_size, byte_offsets=byte_offsets, metadata=metadata, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('metadata', _op.get_attr('metadata'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TFRecordDatasetV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result