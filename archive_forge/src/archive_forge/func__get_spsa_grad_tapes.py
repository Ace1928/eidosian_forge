import warnings
from scipy.linalg import sqrtm
import numpy as np
import pennylane as qml
def _get_spsa_grad_tapes(self, cost, args, kwargs):
    dirs = []
    args_plus = list(args)
    args_minus = list(args)
    for index, arg in enumerate(args):
        if not getattr(arg, 'requires_grad', False):
            continue
        direction = self.rng.choice([-1, 1], size=arg.shape)
        dirs.append(direction)
        args_plus[index] = arg + self.finite_diff_step * direction
        args_minus[index] = arg - self.finite_diff_step * direction
    cost.construct(args_plus, kwargs)
    tape_plus = cost.tape.copy(copy_operations=True)
    cost.construct(args_minus, kwargs)
    tape_minus = cost.tape.copy(copy_operations=True)
    return ([tape_plus, tape_minus], dirs)