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
def tf_record_reader(container: str='', shared_name: str='', compression_type: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """A Reader that outputs the records from a TensorFlow Records file.

  Args:
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    compression_type: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("tf_record_reader op does not support eager execution. Arg 'reader_handle' is a ref.")
    if container is None:
        container = ''
    container = _execute.make_str(container, 'container')
    if shared_name is None:
        shared_name = ''
    shared_name = _execute.make_str(shared_name, 'shared_name')
    if compression_type is None:
        compression_type = ''
    compression_type = _execute.make_str(compression_type, 'compression_type')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TFRecordReader', container=container, shared_name=shared_name, compression_type=compression_type, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('container', _op.get_attr('container'), 'shared_name', _op.get_attr('shared_name'), 'compression_type', _op.get_attr('compression_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TFRecordReader', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result