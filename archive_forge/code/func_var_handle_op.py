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
def var_handle_op(dtype: TV_VarHandleOp_dtype, shape, container: str='', shared_name: str='', allowed_devices=[], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Resource]:
    """Creates a handle to a Variable resource.

  Args:
    dtype: A `tf.DType`. the type of this variable. Must agree with the dtypes
      of all ops using this variable.
    shape: A `tf.TensorShape` or list of `ints`.
      The (possibly partially specified) shape of this variable.
    container: An optional `string`. Defaults to `""`.
      the container this variable is placed in.
    shared_name: An optional `string`. Defaults to `""`.
      the name by which this variable is referred to.
    allowed_devices: An optional list of `strings`. Defaults to `[]`.
      DEPRECATED. The allowed devices containing the resource variable. Set when the
      output ResourceHandle represents a per-replica/partitioned resource variable.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'VarHandleOp', name, 'container', container, 'shared_name', shared_name, 'dtype', dtype, 'shape', shape, 'allowed_devices', allowed_devices)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return var_handle_op_eager_fallback(container=container, shared_name=shared_name, dtype=dtype, shape=shape, allowed_devices=allowed_devices, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    dtype = _execute.make_type(dtype, 'dtype')
    shape = _execute.make_shape(shape, 'shape')
    if container is None:
        container = ''
    container = _execute.make_str(container, 'container')
    if shared_name is None:
        shared_name = ''
    shared_name = _execute.make_str(shared_name, 'shared_name')
    if allowed_devices is None:
        allowed_devices = []
    if not isinstance(allowed_devices, (list, tuple)):
        raise TypeError("Expected list for 'allowed_devices' argument to 'var_handle_op' Op, not %r." % allowed_devices)
    allowed_devices = [_execute.make_str(_s, 'allowed_devices') for _s in allowed_devices]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('VarHandleOp', dtype=dtype, shape=shape, container=container, shared_name=shared_name, allowed_devices=allowed_devices, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('container', _op.get_attr('container'), 'shared_name', _op.get_attr('shared_name'), 'dtype', _op._get_attr_type('dtype'), 'shape', _op.get_attr('shape'), 'allowed_devices', _op.get_attr('allowed_devices'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('VarHandleOp', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result