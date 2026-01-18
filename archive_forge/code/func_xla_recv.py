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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_recv')
def xla_recv(dtype: TV_XlaRecv_dtype, tensor_name: str, shape, name=None) -> _atypes.TensorFuzzingAnnotation[TV_XlaRecv_dtype]:
    """Receives the named tensor from another XLA computation. Wraps the XLA Recv

  operator documented at
   https://www.tensorflow.org/performance/xla/operation_semantics#recv .

  Args:
    dtype: A `tf.DType`. The type of the tensor.
    tensor_name: A `string`. A string key that identifies the channel.
    shape: A `tf.TensorShape` or list of `ints`. The shape of the tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`. The tensor to receive.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaRecv', name, 'dtype', dtype, 'tensor_name', tensor_name, 'shape', shape)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_recv((dtype, tensor_name, shape, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_recv_eager_fallback(dtype=dtype, tensor_name=tensor_name, shape=shape, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_recv, (), dict(dtype=dtype, tensor_name=tensor_name, shape=shape, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_recv((dtype, tensor_name, shape, name), None)
        if _result is not NotImplemented:
            return _result
    dtype = _execute.make_type(dtype, 'dtype')
    tensor_name = _execute.make_str(tensor_name, 'tensor_name')
    shape = _execute.make_shape(shape, 'shape')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaRecv', dtype=dtype, tensor_name=tensor_name, shape=shape, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_recv, (), dict(dtype=dtype, tensor_name=tensor_name, shape=shape, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'tensor_name', _op.get_attr('tensor_name'), 'shape', _op.get_attr('shape'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaRecv', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result