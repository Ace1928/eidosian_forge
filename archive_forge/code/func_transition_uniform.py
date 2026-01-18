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
def transition_uniform(n_states: int) -> np.ndarray:
    """Construct a uniform transition matrix over ``n_states``.

    Parameters
    ----------
    n_states : int > 0
        The number of states

    Returns
    -------
    transition : np.ndarray [shape=(n_states, n_states)]
        ``transition[i, j] = 1./n_states``

    Examples
    --------
    >>> librosa.sequence.transition_uniform(3)
    array([[0.333, 0.333, 0.333],
           [0.333, 0.333, 0.333],
           [0.333, 0.333, 0.333]])
    """
    if not is_positive_int(n_states):
        raise ParameterError(f'n_states={n_states} must be a positive integer')
    transition = np.empty((n_states, n_states), dtype=np.float64)
    transition.fill(1.0 / n_states)
    return transition