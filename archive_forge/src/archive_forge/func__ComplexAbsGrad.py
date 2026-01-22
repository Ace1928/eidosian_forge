import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
@ops.RegisterGradient('ComplexAbs')
def _ComplexAbsGrad(op, grad):
    """Returns the gradient of ComplexAbs."""
    return math_ops.div_no_nan(math_ops.complex(grad, array_ops.zeros_like(grad)) * op.inputs[0], math_ops.complex(op.outputs[0], array_ops.zeros_like(op.outputs[0])))