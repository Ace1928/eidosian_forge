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
def transition_cycle(n_states: int, prob: Union[float, Iterable[float]]) -> np.ndarray:
    """Construct a cyclic transition matrix over ``n_states``.

    The transition matrix will have the following properties:

        - ``transition[i, i] = p``
        - ``transition[i, i + 1] = (1 - p)``

    This type of transition matrix is appropriate for state spaces
    with cyclical structure, such as metrical position within a bar.
    For example, a song in 4/4 time has state transitions of the form

        1->{1, 2}, 2->{2, 3}, 3->{3, 4}, 4->{4, 1}.

    Parameters
    ----------
    n_states : int > 1
        The number of states

    prob : float in [0, 1] or iterable, length=n_states
        If a scalar, this is the probability of a self-transition.

        If a vector of length ``n_states``, ``p[i]`` is the probability of
        self-transition in state ``i``

    Returns
    -------
    transition : np.ndarray [shape=(n_states, n_states)]
        The transition matrix

    Examples
    --------
    >>> librosa.sequence.transition_cycle(4, 0.9)
    array([[0.9, 0.1, 0. , 0. ],
           [0. , 0.9, 0.1, 0. ],
           [0. , 0. , 0.9, 0.1],
           [0.1, 0. , 0. , 0.9]])
    """
    if not (is_positive_int(n_states) and n_states > 1):
        raise ParameterError(f'n_states={n_states} must be a positive integer > 1')
    transition = np.zeros((n_states, n_states), dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    if prob.ndim == 0:
        prob = np.tile(prob, n_states)
    if prob.shape != (n_states,):
        raise ParameterError(f'prob={prob} must have length equal to n_states={n_states}')
    if np.any(prob < 0) or np.any(prob > 1):
        raise ParameterError(f'prob={prob} must have values in the range [0, 1]')
    for i, prob_i in enumerate(prob):
        transition[i, np.mod(i + 1, n_states)] = 1.0 - prob_i
        transition[i, i] = prob_i
    return transition