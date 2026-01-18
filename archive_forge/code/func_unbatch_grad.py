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
def unbatch_grad(original_input: _atypes.TensorFuzzingAnnotation[TV_UnbatchGrad_T], batch_index: _atypes.TensorFuzzingAnnotation[_atypes.Int64], grad: _atypes.TensorFuzzingAnnotation[TV_UnbatchGrad_T], id: _atypes.TensorFuzzingAnnotation[_atypes.Int64], container: str='', shared_name: str='', name=None) -> _atypes.TensorFuzzingAnnotation[TV_UnbatchGrad_T]:
    """Gradient of Unbatch.

  Acts like Batch but using the given batch_index index of batching things as they
  become available. This ensures that the gradients are propagated back in the
  same session which did the forward pass.

  original_input: The input to the Unbatch operation this is the gradient of.
  batch_index: The batch_index given to the Unbatch operation this is the gradient
  of.
  grad: The downstream gradient.
  id: The id scalar emitted by Batch.
  batched_grad: The return value, either an empty tensor or the batched gradient.
  container: Container to control resource sharing.
  shared_name: Instances of UnbatchGrad with the same container and shared_name
   are assumed to possibly belong to the same batch. If left empty, the op name
   will be used as the shared name.

  Args:
    original_input: A `Tensor`.
    batch_index: A `Tensor` of type `int64`.
    grad: A `Tensor`. Must have the same type as `original_input`.
    id: A `Tensor` of type `int64`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `original_input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'UnbatchGrad', name, original_input, batch_index, grad, id, 'container', container, 'shared_name', shared_name)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return unbatch_grad_eager_fallback(original_input, batch_index, grad, id, container=container, shared_name=shared_name, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if container is None:
        container = ''
    container = _execute.make_str(container, 'container')
    if shared_name is None:
        shared_name = ''
    shared_name = _execute.make_str(shared_name, 'shared_name')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('UnbatchGrad', original_input=original_input, batch_index=batch_index, grad=grad, id=id, container=container, shared_name=shared_name, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('container', _op.get_attr('container'), 'shared_name', _op.get_attr('shared_name'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('UnbatchGrad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result