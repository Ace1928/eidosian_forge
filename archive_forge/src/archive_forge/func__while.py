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
def _while(input, cond, body, output_shapes=[], parallel_iterations: int=10, name=None):
    """output = input; While (Cond(output)) { output = Body(output) }

  Args:
    input: A list of `Tensor` objects.
      A list of input tensors whose types are T.
    cond: A function decorated with @Defun.
            A function takes 'input' and returns a tensor.  If the tensor is
            a scalar of non-boolean, the scalar is converted to a boolean
            according to the following rule: if the scalar is a numerical
            value, non-zero means True and zero means False; if the scalar is
            a string, non-empty means True and empty means False. If the
            tensor is not a scalar, non-emptiness means True and False
            otherwise.
    body: A function decorated with @Defun.
            A function that takes a list of tensors and returns another
            list of tensors. Both lists have the same types as specified
            by T.
    output_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
    parallel_iterations: An optional `int`. Defaults to `10`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'While', name, input, 'cond', cond, 'body', body, 'output_shapes', output_shapes, 'parallel_iterations', parallel_iterations)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return _while_eager_fallback(input, cond=cond, body=body, output_shapes=output_shapes, parallel_iterations=parallel_iterations, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if output_shapes is None:
        output_shapes = []
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'while' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if parallel_iterations is None:
        parallel_iterations = 10
    parallel_iterations = _execute.make_int(parallel_iterations, 'parallel_iterations')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('While', input=input, cond=cond, body=body, output_shapes=output_shapes, parallel_iterations=parallel_iterations, name=name)
    _result = _outputs[:]
    if not _result:
        return _op
    if _execute.must_record_gradient():
        _attrs = ('T', _op.get_attr('T'), 'cond', _op.get_attr('cond'), 'body', _op.get_attr('body'), 'output_shapes', _op.get_attr('output_shapes'), 'parallel_iterations', _op._get_attr_int('parallel_iterations'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('While', _inputs_flat, _attrs, _result)
    return _result