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
def __float_window(window_spec):
    """Decorate a window function to support fractional input lengths.

    This function guarantees that for fractional ``x``, the following hold:

    1. ``__float_window(window_function)(x)`` has length ``np.ceil(x)``
    2. all values from ``np.floor(x)`` are set to 0.

    For integer-valued ``x``, there should be no change in behavior.
    """

    def _wrap(n, *args, **kwargs):
        """Wrap the window"""
        n_min, n_max = (int(np.floor(n)), int(np.ceil(n)))
        window = get_window(window_spec, n_min)
        if len(window) < n_max:
            window = np.pad(window, [(0, n_max - len(window))], mode='constant')
        window[n_min:] = 0.0
        return window
    return _wrap