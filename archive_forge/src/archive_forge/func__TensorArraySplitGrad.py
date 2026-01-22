from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
@ops.RegisterGradient('TensorArraySplit')
@ops.RegisterGradient('TensorArraySplitV2')
@ops.RegisterGradient('TensorArraySplitV3')
def _TensorArraySplitGrad(op, flow):
    """Gradient for TensorArraySplit.

  Args:
    op: Forward TensorArraySplit op.
    flow: Gradient `Tensor` flow to TensorArraySplit.

  Returns:
    A grad `Tensor`, the gradient created in upstream ReadGrads or PackGrad.
  """
    handle = op.inputs[0]
    dtype = op.get_attr('T')
    grad_source = _GetGradSource(flow)
    flow_out = array_ops.identity(op.outputs[0], 'flow_out')
    with ops.control_dependencies([flow_out]):
        flow = array_ops.identity(flow, 'write_barrier')
    g = tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow, colocate_with_first_write_call=False).grad(source=grad_source, flow=flow)
    grad = g.concat()
    return [None, grad, None, flow]