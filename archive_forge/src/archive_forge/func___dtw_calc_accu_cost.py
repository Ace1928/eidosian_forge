from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
from .util import pad_center, fill_off_diagonal, is_positive_int, tiny, expand_to
from .util.exceptions import ParameterError
from .filters import get_window
from typing import Any, Iterable, List, Optional, Tuple, Union, overload
from typing_extensions import Literal
from ._typing import _WindowSpec, _IntLike_co
@jit(nopython=True, cache=True)
def __dtw_calc_accu_cost(C: np.ndarray, D: np.ndarray, steps: np.ndarray, step_sizes_sigma: np.ndarray, weights_mul: np.ndarray, weights_add: np.ndarray, max_0: int, max_1: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the accumulated cost matrix D.

    Use dynamic programming to calculate the accumulated costs.

    Parameters
    ----------
    C : np.ndarray [shape=(N, M)]
        pre-computed cost matrix
    D : np.ndarray [shape=(N, M)]
        accumulated cost matrix
    steps : np.ndarray [shape=(N, M)]
        Step matrix, containing the indices of the used steps from the cost
        accumulation step.
    step_sizes_sigma : np.ndarray [shape=[n, 2]]
        Specifies allowed step sizes as used by the dtw.
    weights_add : np.ndarray [shape=[n, ]]
        Additive weights to penalize certain step sizes.
    weights_mul : np.ndarray [shape=[n, ]]
        Multiplicative weights to penalize certain step sizes.
    max_0 : int
        maximum number of steps in step_sizes_sigma in dim 0.
    max_1 : int
        maximum number of steps in step_sizes_sigma in dim 1.

    Returns
    -------
    D : np.ndarray [shape=(N, M)]
        accumulated cost matrix.
        D[N, M] is the total alignment cost.
        When doing subsequence DTW, D[N,:] indicates a matching function.
    steps : np.ndarray [shape=(N, M)]
        Step matrix, containing the indices of the used steps from the cost
        accumulation step.

    See Also
    --------
    dtw
    """
    for cur_n in range(max_0, D.shape[0]):
        for cur_m in range(max_1, D.shape[1]):
            for cur_step_idx, cur_w_add, cur_w_mul in zip(range(step_sizes_sigma.shape[0]), weights_add, weights_mul):
                cur_D = D[cur_n - step_sizes_sigma[cur_step_idx, 0], cur_m - step_sizes_sigma[cur_step_idx, 1]]
                cur_C = cur_w_mul * C[cur_n - max_0, cur_m - max_1]
                cur_C += cur_w_add
                cur_cost = cur_D + cur_C
                if cur_cost < D[cur_n, cur_m]:
                    D[cur_n, cur_m] = cur_cost
                    steps[cur_n, cur_m] = cur_step_idx
    return (D, steps)