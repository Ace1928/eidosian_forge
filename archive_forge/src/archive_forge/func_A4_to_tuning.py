from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def A4_to_tuning(A4: _ScalarOrSequence[_FloatLike_co], *, bins_per_octave: int=12) -> Union[np.floating[Any], np.ndarray]:
    """Convert a reference pitch frequency (e.g., ``A4=435``) to a tuning
    estimation, in fractions of a bin per octave.

    This is useful for determining the tuning deviation relative to
    A440 of a given frequency, assuming equal temperament. By default,
    12 bins per octave are used.

    This method is the inverse of `tuning_to_A4`.

    Examples
    --------
    The base case of this method in which A440 yields 0 tuning offset
    from itself.

    >>> librosa.A4_to_tuning(440.0)
    0.

    Convert a non-A440 frequency to a tuning offset relative
    to A440 using the default of 12 bins per octave.

    >>> librosa.A4_to_tuning(432.0)
    -0.318

    Convert two reference pitch frequencies to corresponding
    tuning estimations at once, but using 24 bins per octave.

    >>> librosa.A4_to_tuning([440.0, 444.0], bins_per_octave=24)
    array([   0.,   0.313   ])

    Parameters
    ----------
    A4 : float or np.ndarray [shape=(n,), dtype=float]
        Reference frequency(s) corresponding to A4.
    bins_per_octave : int > 0
        Number of bins per octave.

    Returns
    -------
    tuning : float or np.ndarray [shape=(n,), dtype=float]
        Tuning deviation from A440 in (fractional) bins per octave.

    See Also
    --------
    tuning_to_A4
    """
    tuning: np.ndarray = bins_per_octave * (np.log2(np.asanyarray(A4)) - np.log2(440.0))
    return tuning