import warnings
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix
from scipy.special import ive  # pylint: disable=no-name-in-module
from ..utils import _cov, _dot, _stack, conditional_jit
def _kde_linear(x, bw='experimental', adaptive=False, extend=False, bound_correction=True, extend_fct=0, bw_fct=1, bw_return=False, custom_lims=None, cumulative=False, grid_len=512, **kwargs):
    """One dimensional density estimation for linear data.

    Given an array of data points `x` it returns an estimate of
    the probability density function that generated the samples in `x`.

    Parameters
    ----------
    x : 1D numpy array
        Data used to calculate the density estimation.
    bw: int, float or str, optional
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth and must be one of "scott",
        "silverman", "isj" or "experimental". Defaults to "experimental".
    adaptive: boolean, optional
        Indicates if the bandwidth is adaptive or not.
        It is the recommended approach when there are multiple modes with different spread.
        It is not compatible with convolution. Defaults to False.
    extend: boolean, optional
        Whether to extend the observed range for `x` in the estimation.
        It extends each bound by a multiple of the standard deviation of `x` given by `extend_fct`.
        Defaults to False.
    bound_correction: boolean, optional
        Whether to perform boundary correction on the bounds of `x` or not.
        Defaults to True.
    extend_fct: float, optional
        Number of standard deviations used to widen the lower and upper bounds of `x`.
        Defaults to 0.5.
    bw_fct: float, optional
        A value that multiplies `bw` which enables tuning smoothness by hand.
        Must be positive. Values below 1 decrease smoothness while values above 1 decrease it.
        Defaults to 1 (no modification).
    bw_return: bool, optional
        Whether to return the estimated bandwidth in addition to the other objects.
        Defaults to False.
    custom_lims: list or tuple, optional
        A list or tuple of length 2 indicating custom bounds for the range of `x`.
        Defaults to None which disables custom bounds.
    cumulative: bool, optional
        Whether return the PDF or the cumulative PDF. Defaults to False.
    grid_len: int, optional
        The number of intervals used to bin the data points i.e. the length of the grid used in
        the estimation. Defaults to 512.

    Returns
    -------
    grid : Gridded numpy array for the x values.
    pdf : Numpy array for the density estimates.
    bw: optional, the estimated bandwidth.
    """
    if not isinstance(bw_fct, (int, float, np.integer, np.floating)):
        raise TypeError(f'`bw_fct` must be a positive number, not an object of {type(bw_fct)}.')
    if bw_fct <= 0:
        raise ValueError(f'`bw_fct` must be a positive number, not {bw_fct}.')
    x_min = x.min()
    x_max = x.max()
    x_std = np.std(x)
    x_range = x_max - x_min
    grid_min, grid_max, grid_len = _get_grid(x_min, x_max, x_std, extend_fct, grid_len, custom_lims, extend, bound_correction)
    grid_counts, _, grid_edges = histogram(x, grid_len, (grid_min, grid_max))
    bw = bw_fct * _get_bw(x, bw, grid_counts, x_std, x_range)
    if adaptive:
        grid, pdf = _kde_adaptive(x, bw, grid_edges, grid_counts, grid_len, bound_correction)
    else:
        grid, pdf = _kde_convolution(x, bw, grid_edges, grid_counts, grid_len, bound_correction)
    if cumulative:
        pdf = pdf.cumsum() / pdf.sum()
    if bw_return:
        return (grid, pdf, bw)
    else:
        return (grid, pdf)