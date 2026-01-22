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
def __dtw_backtracking(steps: np.ndarray, step_sizes_sigma: np.ndarray, subseq: bool, start: Optional[int]=None) -> List[Tuple[int, int]]:
    """Backtrack optimal warping path.

    Uses the saved step sizes from the cost accumulation
    step to backtrack the index pairs for an optimal
    warping path.

    Parameters
    ----------
    steps : np.ndarray [shape=(N, M)]
        Step matrix, containing the indices of the used steps from the cost
        accumulation step.
    step_sizes_sigma : np.ndarray [shape=[n, 2]]
        Specifies allowed step sizes as used by the dtw.
    subseq : bool
        Enable subsequence DTW, e.g., for retrieval tasks.
    start : int
        Start column index for backtraing (only allowed for ``subseq=True``)

    Returns
    -------
    wp : list [shape=(N,)]
        Warping path with index pairs.
        Each list entry contains an index pair
        (n, m) as a tuple

    See Also
    --------
    dtw
    """
    if start is None:
        cur_idx = (steps.shape[0] - 1, steps.shape[1] - 1)
    else:
        cur_idx = (steps.shape[0] - 1, start)
    wp = []
    wp.append((cur_idx[0], cur_idx[1]))
    while subseq and cur_idx[0] > 0 or (not subseq and cur_idx != (0, 0)):
        cur_step_idx = steps[cur_idx[0], cur_idx[1]]
        cur_idx = (cur_idx[0] - step_sizes_sigma[cur_step_idx][0], cur_idx[1] - step_sizes_sigma[cur_step_idx][1])
        if min(cur_idx) < 0:
            break
        wp.append((cur_idx[0], cur_idx[1]))
    return wp