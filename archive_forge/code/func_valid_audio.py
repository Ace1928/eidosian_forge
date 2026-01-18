from __future__ import annotations
import scipy.ndimage
import scipy.sparse
import numpy as np
import numba
from numpy.lib.stride_tricks import as_strided
from .._cache import cache
from .exceptions import ParameterError
from .deprecation import Deprecated
from numpy.typing import ArrayLike, DTypeLike
from typing import (
from typing_extensions import Literal
from .._typing import _SequenceLike, _FloatLike_co, _ComplexLike_co
@cache(level=20)
def valid_audio(y: np.ndarray, *, mono: Union[bool, Deprecated]=Deprecated()) -> bool:
    """Determine whether a variable contains valid audio data.

    The following conditions must be satisfied:

    - ``type(y)`` is ``np.ndarray``
    - ``y.dtype`` is floating-point
    - ``y.ndim != 0`` (must have at least one dimension)
    - ``np.isfinite(y).all()`` samples must be all finite values

    If ``mono`` is specified, then we additionally require
    - ``y.ndim == 1``

    Parameters
    ----------
    y : np.ndarray
        The input data to validate

    mono : bool
        Whether or not to require monophonic audio

        .. warning:: The ``mono`` parameter is deprecated in version 0.9 and will be
          removed in 0.10.

    Returns
    -------
    valid : bool
        True if all tests pass

    Raises
    ------
    ParameterError
        In any of the conditions specified above fails

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    >>> # By default, valid_audio allows only mono signals
    >>> filepath = librosa.ex('trumpet', hq=True)
    >>> y_mono, sr = librosa.load(filepath, mono=True)
    >>> y_stereo, _ = librosa.load(filepath, mono=False)
    >>> librosa.util.valid_audio(y_mono), librosa.util.valid_audio(y_stereo)
    True, False

    >>> # To allow stereo signals, set mono=False
    >>> librosa.util.valid_audio(y_stereo, mono=False)
    True

    See Also
    --------
    numpy.float32
    """
    if not isinstance(y, np.ndarray):
        raise ParameterError('Audio data must be of type numpy.ndarray')
    if not np.issubdtype(y.dtype, np.floating):
        raise ParameterError('Audio data must be floating-point')
    if y.ndim == 0:
        raise ParameterError(f'Audio data must be at least one-dimensional, given y.shape={y.shape}')
    if isinstance(mono, Deprecated):
        mono = False
    if mono and y.ndim != 1:
        raise ParameterError(f'Invalid shape for monophonic audio: ndim={y.ndim:d}, shape={y.shape}')
    if not np.isfinite(y).all():
        raise ParameterError('Audio buffer is not finite everywhere')
    return True