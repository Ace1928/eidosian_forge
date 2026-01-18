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
def transition_local(n_states: int, width: Union[int, Iterable[int]], *, window: _WindowSpec='triangle', wrap: bool=False) -> np.ndarray:
    """Construct a localized transition matrix.

    The transition matrix will have the following properties:

        - ``transition[i, j] = 0`` if ``|i - j| > width``
        - ``transition[i, i]`` is maximal
        - ``transition[i, i - width//2 : i + width//2]`` has shape ``window``

    This type of transition matrix is appropriate for state spaces
    that discretely approximate continuous variables, such as in fundamental
    frequency estimation.

    Parameters
    ----------
    n_states : int > 1
        The number of states

    width : int >= 1 or iterable
        The maximum number of states to treat as "local".
        If iterable, it should have length equal to ``n_states``,
        and specify the width independently for each state.

    window : str, callable, or window specification
        The window function to determine the shape of the "local" distribution.

        Any window specification supported by `filters.get_window` will work here.

        .. note:: Certain windows (e.g., 'hann') are identically 0 at the boundaries,
            so and effectively have ``width-2`` non-zero values.  You may have to expand
            ``width`` to get the desired behavior.

    wrap : bool
        If ``True``, then state locality ``|i - j|`` is computed modulo ``n_states``.
        If ``False`` (default), then locality is absolute.

    See Also
    --------
    librosa.filters.get_window

    Returns
    -------
    transition : np.ndarray [shape=(n_states, n_states)]
        The transition matrix

    Examples
    --------
    Triangular distributions with and without wrapping

    >>> librosa.sequence.transition_local(5, 3, window='triangle', wrap=False)
    array([[0.667, 0.333, 0.   , 0.   , 0.   ],
           [0.25 , 0.5  , 0.25 , 0.   , 0.   ],
           [0.   , 0.25 , 0.5  , 0.25 , 0.   ],
           [0.   , 0.   , 0.25 , 0.5  , 0.25 ],
           [0.   , 0.   , 0.   , 0.333, 0.667]])

    >>> librosa.sequence.transition_local(5, 3, window='triangle', wrap=True)
    array([[0.5 , 0.25, 0.  , 0.  , 0.25],
           [0.25, 0.5 , 0.25, 0.  , 0.  ],
           [0.  , 0.25, 0.5 , 0.25, 0.  ],
           [0.  , 0.  , 0.25, 0.5 , 0.25],
           [0.25, 0.  , 0.  , 0.25, 0.5 ]])

    Uniform local distributions with variable widths and no wrapping

    >>> librosa.sequence.transition_local(5, [1, 2, 3, 3, 1], window='ones', wrap=False)
    array([[1.   , 0.   , 0.   , 0.   , 0.   ],
           [0.5  , 0.5  , 0.   , 0.   , 0.   ],
           [0.   , 0.333, 0.333, 0.333, 0.   ],
           [0.   , 0.   , 0.333, 0.333, 0.333],
           [0.   , 0.   , 0.   , 0.   , 1.   ]])
    """
    if not (is_positive_int(n_states) and n_states > 1):
        raise ParameterError(f'n_states={n_states} must be a positive integer > 1')
    width = np.asarray(width, dtype=int)
    if width.ndim == 0:
        width = np.tile(width, n_states)
    if width.shape != (n_states,):
        raise ParameterError(f'width={width} must have length equal to n_states={n_states}')
    if np.any(width < 1):
        raise ParameterError(f'width={width} must be at least 1')
    transition = np.zeros((n_states, n_states), dtype=np.float64)
    for i, width_i in enumerate(width):
        trans_row = pad_center(get_window(window, width_i, fftbins=False), size=n_states)
        trans_row = np.roll(trans_row, n_states // 2 + i + 1)
        if not wrap:
            trans_row[min(n_states, i + width_i // 2 + 1):] = 0
            trans_row[:max(0, i - width_i // 2)] = 0
        transition[i] = trans_row
    transition /= transition.sum(axis=1, keepdims=True)
    return transition