import warnings
import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
from numba import jit
from ._cache import cache
from . import util
from .util.exceptions import ParameterError
from .util.decorators import deprecated
from .core.convert import note_to_hz, hz_to_midi, midi_to_hz, hz_to_octs
from .core.convert import fft_frequencies, mel_frequencies
from numpy.typing import ArrayLike, DTypeLike
from typing import Any, List, Optional, Tuple, Union
from typing_extensions import Literal
from ._typing import _WindowSpec, _FloatLike_co
@cache(level=10)
def wavelet_lengths(*, freqs: ArrayLike, sr: float=22050, window: _WindowSpec='hann', filter_scale: float=1, gamma: Optional[float]=0, alpha: Optional[Union[float, np.ndarray]]=None) -> Tuple[np.ndarray, float]:
    """Return length of each filter in a wavelet basis.

    Parameters
    ----------
    freqs : np.ndarray (positive)
        Center frequencies of the filters (in Hz).
        Must be in ascending order.

    sr : number > 0 [scalar]
        Audio sampling rate

    window : str or callable
        Window function to use on filters

    filter_scale : float > 0 [scalar]
        Resolution of filter windows. Larger values use longer windows.

    gamma : number >= 0 [scalar, optional]
        Bandwidth offset for determining filter lengths, as used in
        Variable-Q transforms.

        Bandwidth for the k'th filter is determined by::

            B[k] = alpha[k] * freqs[k] + gamma

        ``alpha[k]`` is twice the relative difference between ``freqs[k+1]`` and ``freqs[k-1]``::

            alpha[k] = (freqs[k+1]-freqs[k-1]) / (freqs[k+1]+freqs[k-1])

        If ``freqs`` follows a geometric progression (as in CQT and VQT), the vector
        ``alpha`` is constant and such that::

            (1 + alpha) * freqs[k-1] = (1 - alpha) * freqs[k+1]

        Furthermore, if ``gamma=0`` (default), ``alpha`` is such that even-``k`` and
        odd-``k`` filters are interleaved::

            freqs[k-1] + B[k-1] = freqs[k+1] - B[k+1]

        If ``gamma=None`` is specified, then ``gamma`` is computed such
        that each filter has bandwidth proportional to the equivalent
        rectangular bandwidth (ERB) at frequency ``freqs[k]``::

            gamma[k] = 24.7 * alpha[k] / 0.108

        as derived by [#]_.

        .. [#] Glasberg, Brian R., and Brian CJ Moore.
            "Derivation of auditory filter shapes from notched-noise data."
            Hearing research 47.1-2 (1990): 103-138.

    alpha : number > 0 [optional]
        Optional pre-computed relative bandwidth parameter.
        Note that this must be provided if ``len(freqs)==1`` because bandwidth cannot be
        inferred from a single frequency.
        Otherwise, if left unspecified, it will be automatically derived by the rules
        specified above.

    Returns
    -------
    lengths : np.ndarray
        The length of each filter.
    f_cutoff : float
        The lowest frequency at which all filters' main lobes have decayed by
        at least 3dB.

        This second output serves in cqt and vqt to ensure that all wavelet
        bands remain below the Nyquist frequency.

    Notes
    -----
    This function caches at level 10.

    Raises
    ------
    ParameterError
        - If ``filter_scale`` is not strictly positive

        - If ``gamma`` is a negative number

        - If any frequencies are <= 0

        - If the frequency array is not sorted in ascending order
    """
    freqs = np.asarray(freqs)
    if filter_scale <= 0:
        raise ParameterError(f'filter_scale={filter_scale} must be positive')
    if gamma is not None and gamma < 0:
        raise ParameterError(f'gamma={gamma} must be non-negative')
    if np.any(freqs <= 0):
        raise ParameterError('frequencies must be strictly positive')
    if len(freqs) > 1 and np.any(freqs[:-1] > freqs[1:]):
        raise ParameterError(f'Frequency array={freqs} must be in strictly ascending order')
    if alpha is None:
        alpha = _relative_bandwidth(freqs=freqs)
    else:
        alpha = np.asarray(alpha)
    gamma_: Union[_FloatLike_co, np.ndarray]
    if gamma is None:
        gamma_ = alpha * 24.7 / 0.108
    else:
        gamma_ = gamma
    Q = float(filter_scale) / alpha
    f_cutoff = max(freqs * (1 + 0.5 * window_bandwidth(window) / Q) + 0.5 * gamma_)
    lengths = Q * sr / (freqs + gamma_ / alpha)
    return (lengths, f_cutoff)