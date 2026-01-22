import numpy as np
from onnx.reference.ops.aionnx_preview_training._op_run_training import OpRunTraining
def _apply_adagrad(r, t, x, g, h, norm_coefficient, epsilon, decay_factor):
    r_ = r / (1 + t * decay_factor)
    g_regularized = norm_coefficient * x + g
    h_new = h + g_regularized * g_regularized
    h_sqrt = np.sqrt(h_new) + epsilon
    x_new = x - r_ * g_regularized / h_sqrt
    return (x_new, h_new)