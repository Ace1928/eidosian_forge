import warnings
from scipy.linalg import sqrtm
import numpy as np
import pennylane as qml
def _step_core(self, cost, args, kwargs):
    """Core step function that returns the updated parameter before blocking condition
        is applied.

        Args:
            cost (QNode): the QNode wrapper for the objective function for optimization
            args : variable length argument list for qnode
            kwargs : variable length of keyword arguments for the qnode

        Returns:
            np.array: the new variable values :math:`x^{(t+1)}` before the blocking condition
            is applied.
        """
    all_grad_tapes = []
    all_grad_dirs = []
    all_metric_tapes = []
    all_tensor_dirs = []
    for _ in range(self.resamplings):
        grad_tapes, grad_dirs = self._get_spsa_grad_tapes(cost, args, kwargs)
        metric_tapes, tensor_dirs = self._get_tensor_tapes(cost, args, kwargs)
        all_grad_tapes += grad_tapes
        all_metric_tapes += metric_tapes
        all_grad_dirs.append(grad_dirs)
        all_tensor_dirs.append(tensor_dirs)
    if isinstance(cost.device, qml.devices.Device):
        program, config = cost.device.preprocess()
        raw_results = qml.execute(all_grad_tapes + all_metric_tapes, cost.device, None, transform_program=program, config=config)
    else:
        raw_results = qml.execute(all_grad_tapes + all_metric_tapes, cost.device, None)
    grads = [self._post_process_grad(raw_results[2 * i:2 * i + 2], all_grad_dirs[i]) for i in range(self.resamplings)]
    grads = np.array(grads)
    metric_tensors = [self._post_process_tensor(raw_results[2 * self.resamplings + 4 * i:2 * self.resamplings + 4 * i + 4], all_tensor_dirs[i]) for i in range(self.resamplings)]
    metric_tensors = np.array(metric_tensors)
    grad_avg = np.mean(grads, axis=0)
    tensor_avg = np.mean(metric_tensors, axis=0)
    self._update_tensor(tensor_avg)
    params_next = self._get_next_params(args, grad_avg)
    return params_next[0] if len(params_next) == 1 else params_next