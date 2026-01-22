import warnings
from scipy.linalg import sqrtm
import numpy as np
import pennylane as qml
def _get_tensor_tapes(self, cost, args, kwargs):
    dir1_list = []
    dir2_list = []
    args_list = [list(args) for _ in range(4)]
    for index, arg in enumerate(args):
        if not getattr(arg, 'requires_grad', False):
            continue
        dir1 = self.rng.choice([-1, 1], size=arg.shape)
        dir2 = self.rng.choice([-1, 1], size=arg.shape)
        dir1_list.append(dir1.reshape(-1))
        dir2_list.append(dir2.reshape(-1))
        args_list[0][index] = arg + self.finite_diff_step * (dir1 + dir2)
        args_list[1][index] = arg + self.finite_diff_step * dir1
        args_list[2][index] = arg + self.finite_diff_step * (-dir1 + dir2)
        args_list[3][index] = arg - self.finite_diff_step * dir1
    dir_vecs = (np.concatenate(dir1_list), np.concatenate(dir2_list))
    tapes = [self._get_overlap_tape(cost, args, args_finite_diff, kwargs) for args_finite_diff in args_list]
    return (tapes, dir_vecs)