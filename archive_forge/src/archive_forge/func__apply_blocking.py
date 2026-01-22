import warnings
from scipy.linalg import sqrtm
import numpy as np
import pennylane as qml
def _apply_blocking(self, cost, args, kwargs, params_next):
    cost.construct(args, kwargs)
    tape_loss_curr = cost.tape.copy(copy_operations=True)
    if not isinstance(params_next, list):
        params_next = [params_next]
    cost.construct(params_next, kwargs)
    tape_loss_next = cost.tape.copy(copy_operations=True)
    program, _ = cost.device.preprocess()
    loss_curr, loss_next = qml.execute([tape_loss_curr, tape_loss_next], cost.device, None, transform_program=program)
    ind = (self.k - 2) % self.last_n_steps.size
    self.last_n_steps[ind] = loss_curr
    tol = 2 * self.last_n_steps.std() if self.k > self.last_n_steps.size else 2 * self.last_n_steps[:self.k - 1].std()
    if loss_curr + tol < loss_next:
        params_next = args
    if len(params_next) == 1:
        return (params_next[0], loss_curr)
    return (params_next, loss_curr)