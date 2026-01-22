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
@ops.RegisterGradient('CumulativeLogsumexp')
def _CumulativeLogsumexpGrad(op, grad):
    x = op.inputs[0]
    axis = op.inputs[1]
    cumulative_logsumexp = op.outputs[0]
    exclusive = op.get_attr('exclusive')
    reverse = op.get_attr('reverse')
    log_grad_positive = array_ops.where_v2(math_ops.greater(grad, 0), math_ops.log(grad), grad.dtype.min)
    log_grad_negative = array_ops.where_v2(math_ops.less(grad, 0), math_ops.log(-grad), grad.dtype.min)
    output_pos = math_ops.exp(math_ops.cumulative_logsumexp(log_grad_positive - cumulative_logsumexp, axis=axis, reverse=not reverse, exclusive=exclusive) + x)
    output_neg = math_ops.exp(math_ops.cumulative_logsumexp(log_grad_negative - cumulative_logsumexp, axis=axis, reverse=not reverse, exclusive=exclusive) + x)
    return [output_pos - output_neg, None]