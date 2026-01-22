import numpy as np
from scipy import fft as sp_fft
from . import _signaltools
from .windows import get_window
from ._spectral import _lombscargle
from ._arraytools import const_ext, even_ext, odd_ext, zero_ext
import warnings

    Returns the bias of the median of a set of periodograms relative to
    the mean.

    See Appendix B from [1]_ for details.

    Parameters
    ----------
    n : int
        Numbers of periodograms being averaged.

    Returns
    -------
    bias : float
        Calculated bias.

    References
    ----------
    .. [1] B. Allen, W.G. Anderson, P.R. Brady, D.A. Brown, J.D.E. Creighton.
           "FINDCHIRP: an algorithm for detection of gravitational waves from
           inspiraling compact binaries", Physical Review D 85, 2012,
           :arxiv:`gr-qc/0509116`
    