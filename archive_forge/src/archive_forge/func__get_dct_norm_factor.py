import math
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy.fft._fft import _cook_shape
from cupyx.scipy.fft import _fft
def _get_dct_norm_factor(n, inorm, dct_type=2):
    """Normalization factors for DCT/DST I-IV.

    Parameters
    ----------
    n : int
        Data size.
    inorm : {'none', 'sqrt', 'full'}
        When `inorm` is 'none', the scaling factor is 1.0 (unnormalized). When
        `inorm` is 1, scaling by ``1/sqrt(d)`` as needed for an orthogonal
        transform is used. When `inorm` is 2, normalization by ``1/d`` is
        applied. The value of ``d`` depends on both `n` and the `dct_type`.
    dct_type : {1, 2, 3, 4}
        Which type of DCT or DST is being normalized?.

    Returns
    -------
    fct : float
        The normalization factor.
    """
    if inorm == 'none':
        return 1
    delta = -1 if dct_type == 1 else 0
    d = 2 * (n + delta)
    if inorm == 'full':
        fct = 1 / d
    elif inorm == 'sqrt':
        fct = 1 / math.sqrt(d)
    else:
        raise ValueError('expected inorm = "none", "sqrt" or "full"')
    return fct