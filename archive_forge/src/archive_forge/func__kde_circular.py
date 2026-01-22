import warnings
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix
from scipy.special import ive  # pylint: disable=no-name-in-module
from ..utils import _cov, _dot, _stack, conditional_jit
def _kde_circular(x, bw='taylor', bw_fct=1, bw_return=False, custom_lims=None, cumulative=False, grid_len=512, **kwargs):
    """One dimensional density estimation for circular data.

    Given an array of data points `x` measured in radians, it returns an estimate of the
    probability density function that generated the samples in `x`.

    Parameters
    ----------
    x : 1D numpy array
        Data used to calculate the density estimation.
    bw: int, float or str, optional
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth and must be "taylor" since it is the
        only option supported so far. Defaults to "taylor".
    bw_fct: float, optional
        A value that multiplies `bw` which enables tuning smoothness by hand. Must be positive.
        Values above 1 decrease smoothness while values below 1 decrease it.
        Defaults to 1 (no modification).
    bw_return: bool, optional
        Whether to return the estimated bandwidth in addition to the other objects.
        Defaults to False.
    custom_lims: list or tuple, optional
        A list or tuple of length 2 indicating custom bounds for the range of `x`.
        Defaults to None which means the estimation limits are [-pi, pi].
    cumulative: bool, optional
        Whether return the PDF or the cumulative PDF. Defaults to False.
    grid_len: int, optional
        The number of intervals used to bin the data pointa i.e. the length of the grid used in the
        estimation. Defaults to 512.
    """
    x = _normalize_angle(x)
    if not isinstance(bw_fct, (int, float, np.integer, np.floating)):
        raise TypeError(f'`bw_fct` must be a positive number, not an object of {type(bw_fct)}.')
    if bw_fct <= 0:
        raise ValueError(f'`bw_fct` must be a positive number, not {bw_fct}.')
    if isinstance(bw, bool):
        raise ValueError("`bw` can't be of type `bool`.\nExpected a positive numeric or 'taylor'")
    if isinstance(bw, (int, float)) and bw < 0:
        raise ValueError(f'Numeric `bw` must be positive.\nInput: {bw:.4f}.')
    if isinstance(bw, str):
        if bw == 'taylor':
            bw = _bw_taylor(x)
        else:
            raise ValueError(f'`bw` must be a positive numeric or `taylor`, not {bw}')
    bw *= bw_fct
    if custom_lims is not None:
        custom_lims = _check_custom_lims(custom_lims, x.min(), x.max())
        grid_min = custom_lims[0]
        grid_max = custom_lims[1]
        assert grid_min >= -np.pi, "Lower limit can't be smaller than -pi"
        assert grid_max <= np.pi, "Upper limit can't be larger than pi"
    else:
        grid_min = -np.pi
        grid_max = np.pi
    bins = np.linspace(grid_min, grid_max, grid_len + 1)
    bin_counts, _, bin_edges = histogram(x, bins=bins)
    grid = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    kern = _vonmises_pdf(x=grid, mu=0, kappa=bw)
    pdf = np.fft.fftshift(np.fft.irfft(np.fft.rfft(kern) * np.fft.rfft(bin_counts)))
    pdf /= len(x)
    if cumulative:
        pdf = pdf.cumsum() / pdf.sum()
    if bw_return:
        return (grid, pdf, bw)
    else:
        return (grid, pdf)