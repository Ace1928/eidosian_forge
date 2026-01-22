import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
def _log_ndtr_lower(x, series_order):
    """Asymptotic expansion version of `Log[cdf(x)]`, appropriate for `x<<-1`."""
    x_2 = math_ops.square(x)
    log_scale = -0.5 * x_2 - math_ops.log(-x) - 0.5 * np.log(2.0 * np.pi)
    return log_scale + math_ops.log(_log_ndtr_asymptotic_series(x, series_order))