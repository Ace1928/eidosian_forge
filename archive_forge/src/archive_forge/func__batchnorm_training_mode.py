import numpy as np
from onnx.reference.op_run import OpRun
def _batchnorm_training_mode(x: np.ndarray, s: np.ndarray, bias: np.ndarray, mean: np.ndarray, var: np.ndarray, momentum: float=0.9, epsilon: float=1e-05) -> np.ndarray:
    axis = tuple(np.delete(np.arange(len(x.shape)), 1))
    saved_mean = x.mean(axis=axis)
    saved_var = x.var(axis=axis)
    output_mean = mean * momentum + saved_mean * (1 - momentum)
    output_var = var * momentum + saved_var * (1 - momentum)
    y = _batchnorm_test_mode(x, s, bias, saved_mean, saved_var, epsilon=epsilon)
    return (y.astype(x.dtype), saved_mean.astype(x.dtype), saved_var.astype(x.dtype), output_mean.astype(x.dtype), output_var.astype(x.dtype))