from tensorflow.python.framework import ops
from tensorflow.python.ops import manip_ops
@ops.RegisterGradient('Roll')
def _RollGrad(op, grad):
    shift = op.inputs[1]
    axis = op.inputs[2]
    roll_grad = manip_ops.roll(grad, -shift, axis)
    return (roll_grad, None, None)