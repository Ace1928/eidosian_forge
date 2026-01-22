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
def _relative_bandwidth(*, freqs: np.ndarray) -> np.ndarray:
    """Compute the relative bandwidth for each of a set of specified frequencies.

    This function is used as a helper in wavelet basis construction.

    Parameters
    ----------
    freqs : np.ndarray
        The array of frequencies

    Returns
    -------
    alpha : np.ndarray
        Relative bandwidth
    """
    if len(freqs) <= 1:
        raise ParameterError(f'2 or more frequencies are required to compute bandwidths. Given freqs={freqs}')
    bpo = np.empty_like(freqs)
    logf = np.log2(freqs)
    bpo[0] = 1 / (logf[1] - logf[0])
    bpo[-1] = 1 / (logf[-1] - logf[-2])
    bpo[1:-1] = 2 / (logf[2:] - logf[:-2])
    alpha = (2.0 ** (2 / bpo) - 1) / (2.0 ** (2 / bpo) + 1)
    return alpha