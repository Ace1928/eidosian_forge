import warnings
from scipy.linalg import sqrtm
import numpy as np
import pennylane as qml
def _get_next_params(self, args, gradient):
    params = []
    non_trainable_indices = []
    for idx, arg in enumerate(args):
        if not getattr(arg, 'requires_grad', False):
            non_trainable_indices.append(idx)
            continue
        if arg.shape == ():
            arg = arg.reshape(-1)
        params.append(arg)
    params_vec = np.concatenate([param.reshape(-1) for param in params])
    grad_vec = np.concatenate([grad.reshape(-1) for grad in gradient])
    new_params_vec = np.linalg.solve(self.metric_tensor, -self.stepsize * grad_vec + np.matmul(self.metric_tensor, params_vec))
    params_split_indices = []
    tmp = 0
    for param in params:
        tmp += param.size
        params_split_indices.append(tmp)
    new_params = np.split(new_params_vec, params_split_indices)
    new_params_reshaped = [new_params[i].reshape(params[i].shape) for i in range(len(params))]
    next_args = []
    non_trainable_idx = 0
    trainable_idx = 0
    for idx, arg in enumerate(args):
        if non_trainable_idx < len(non_trainable_indices) and idx == non_trainable_indices[non_trainable_idx]:
            next_args.append(arg)
            non_trainable_idx += 1
            continue
        next_args.append(new_params_reshaped[trainable_idx])
        trainable_idx += 1
    return next_args