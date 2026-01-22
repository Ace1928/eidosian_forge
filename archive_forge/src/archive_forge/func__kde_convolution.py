import warnings
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix
from scipy.special import ive  # pylint: disable=no-name-in-module
from ..utils import _cov, _dot, _stack, conditional_jit
def _kde_convolution(x, bw, grid_edges, grid_counts, grid_len, bound_correction, **kwargs):
    """Kernel density with convolution.

    One dimensional Gaussian kernel density estimation via convolution of the binned relative
    frequencies and a Gaussian filter. This is an internal function used by `kde()`.
    """
    bin_width = grid_edges[1] - grid_edges[0]
    f = grid_counts / bin_width / len(x)
    bw /= bin_width
    grid = (grid_edges[1:] + grid_edges[:-1]) / 2
    kernel_n = int(bw * 2 * np.pi)
    if kernel_n == 0:
        kernel_n = 1
    kernel = gaussian(kernel_n, bw)
    if bound_correction:
        npad = int(grid_len / 5)
        f = np.concatenate([f[npad - 1::-1], f, f[grid_len:grid_len - npad - 1:-1]])
        pdf = convolve(f, kernel, mode='same', method='direct')[npad:npad + grid_len]
    else:
        pdf = convolve(f, kernel, mode='same', method='direct')
    pdf /= bw * (2 * np.pi) ** 0.5
    return (grid, pdf)