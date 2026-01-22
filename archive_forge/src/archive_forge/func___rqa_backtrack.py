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
def __rqa_backtrack(score, pointers):
    """RQA path backtracking

    Given the score matrix and backtracking index array,
    reconstruct the optimal path.
    """
    offsets = [(-1, -1), (-1, -2), (-2, -1)]
    idx = list(np.unravel_index(np.argmax(score), score.shape))
    path: List = []
    while True:
        bt_index = pointers[tuple(idx)]
        if bt_index == -1:
            break
        path.insert(0, idx)
        if bt_index == -2:
            break
        idx = [idx[_] + offsets[bt_index][_] for _ in range(len(idx))]
    if not path:
        return np.empty((0, 2), dtype=np.uint)
    return np.asarray(path, dtype=np.uint)