import re
import numpy as np
from numba import jit
from .intervals import INTERVALS
from .._cache import cache
from ..util.exceptions import ParameterError
from typing import Dict, List, Union, overload
from ..util.decorators import vectorize
from .._typing import _ScalarOrSequence, _FloatLike_co, _SequenceLike
Convert an interval to Functional Just System (FJS) notation.

    See https://misotanni.github.io/fjs/en/index.html for a thorough overview
    of the FJS notation system, and the examples below.

    FJS conversion works by identifying a Pythagorean interval which is within
    a specified tolerance of the target interval, which provides the core note
    name.  If the interval is derived from ratios other than perfect fifths,
    then the remaining factors are encoded as superscripts for otonal
    (increasing) intervals and subscripts for utonal (decreasing) intervals.

    Parameters
    ----------
    interval : float > 0 or iterable of floats
        A (just) interval to notate in FJS.

    unison : str
        The name of the unison note (corresponding to `interval=1`).

    tolerance : float
        The tolerance threshold for identifying the core note name.

    unicode : bool
        If ``True`` (default), use Unicode symbols (♯𝄪♭𝄫)for accidentals,
        and superscripts/subscripts for otonal and utonal accidentals.

        If ``False``, accidentals will be encoded as low-order ASCII representations::

            ♯ -> #, 𝄪 -> ##, ♭ -> b, 𝄫 -> bb

        Otonal and utonal accidentals will be denoted by `^##` and `_##`
        respectively (see examples below).

    Raises
    ------
    ParameterError
        If the provided interval is not positive

        If the provided interval cannot be identified with a
        just intonation prime factorization.

    Returns
    -------
    note_fjs : str or np.ndarray(dtype=str)
        The interval(s) relative to the given unison in FJS notation.

    Examples
    --------
    Pythagorean intervals appear as expected, with no otonal
    or utonal extensions:

    >>> librosa.interval_to_fjs(3/2, unison='C')
    'G'
    >>> librosa.interval_to_fjs(4/3, unison='F')
    'B♭'

    A ptolemaic major third will appear with an otonal '5':

    >>> librosa.interval_to_fjs(5/4, unison='A')
    'C♯⁵'

    And a ptolemaic minor third will appear with utonal '5':

    >>> librosa.interval_to_fjs(6/5, unison='A')
    'C₅'

    More complex intervals will have compound accidentals.
    For example:

    >>> librosa.interval_to_fjs(25/14, unison='F#')
    'E²⁵₇'
    >>> librosa.interval_to_fjs(25/14, unison='F#', unicode=False)
    'E^25_7'

    Array inputs are also supported:

    >>> librosa.interval_to_fjs([3/2, 4/3, 5/3])
    array(['G', 'F', 'A⁵'], dtype='<U2')

    