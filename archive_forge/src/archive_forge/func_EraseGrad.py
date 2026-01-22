from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_map_ops
from tensorflow.python.ops.gen_map_ops import *
@ops.RegisterGradient('TensorMapErase')
def EraseGrad(op, dmap):
    key_grad = None
    map_grad = dmap
    return (map_grad, key_grad)