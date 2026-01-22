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
def anonymous_multi_device_iterator(devices, output_types, output_shapes, name=None):
    """A container for a multi device iterator resource.

  Args:
    devices: A list of `strings` that has length `>= 1`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (handle, deleter).

    handle: A `Tensor` of type `resource`.
    deleter: A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'AnonymousMultiDeviceIterator', name, 'devices', devices, 'output_types', output_types, 'output_shapes', output_shapes)
            _result = _AnonymousMultiDeviceIteratorOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return anonymous_multi_device_iterator_eager_fallback(devices=devices, output_types=output_types, output_shapes=output_shapes, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(devices, (list, tuple)):
        raise TypeError("Expected list for 'devices' argument to 'anonymous_multi_device_iterator' Op, not %r." % devices)
    devices = [_execute.make_str(_s, 'devices') for _s in devices]
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'anonymous_multi_device_iterator' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'anonymous_multi_device_iterator' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('AnonymousMultiDeviceIterator', devices=devices, output_types=output_types, output_shapes=output_shapes, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('devices', _op.get_attr('devices'), 'output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('AnonymousMultiDeviceIterator', _inputs_flat, _attrs, _result)
    _result = _AnonymousMultiDeviceIteratorOutput._make(_result)
    return _result