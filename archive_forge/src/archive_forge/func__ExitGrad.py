from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.control_flow_ops import *
@ops.RegisterGradient('Exit')
def _ExitGrad(op, grad):
    """Gradients for an exit op are calculated using an Enter op."""
    graph = ops.get_default_graph()
    op_ctxt = op._get_control_flow_context()
    grad_ctxt = graph._get_control_flow_context()
    if not grad_ctxt.back_prop:
        return None
    if op_ctxt.grad_state:
        raise TypeError('Second-order gradient for while loops not supported.')
    if isinstance(grad, tensor.Tensor):
        grad_ctxt.AddName(grad.name)
    else:
        if not isinstance(grad, (indexed_slices.IndexedSlices, sparse_tensor.SparseTensor)):
            raise TypeError(f'Type {type(grad)} not supported, must be either`indexed_slices.IndexedSlices` or `SparseTensor`.')
        grad_ctxt.AddName(grad.values.name)
        grad_ctxt.AddName(grad.indices.name)
        dense_shape = grad.dense_shape
        if dense_shape is not None:
            grad_ctxt.AddName(dense_shape.name)
    grad_ctxt.Enter()
    result = control_flow_ops._Enter(grad, grad_ctxt.name, is_constant=False, parallel_iterations=grad_ctxt.parallel_iterations, name='b_exit')
    grad_ctxt.loop_enters.append(result)
    grad_ctxt.Exit()
    return result