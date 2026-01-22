import warnings
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix
from scipy.special import ive  # pylint: disable=no-name-in-module
from ..utils import _cov, _dot, _stack, conditional_jit
def _normalize_angle(x, zero_centered=True):
    """Normalize angles.

    Normalize angles in radians to [-pi, pi) or [0, 2 * pi) according to `zero_centered`.
    """
    if zero_centered:
        return (x + np.pi) % (2 * np.pi) - np.pi
    else:
        return x % (2 * np.pi)