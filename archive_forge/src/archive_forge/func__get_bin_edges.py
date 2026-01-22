import operator
import warnings
import numpy
import cupy
from cupy import _core
from cupy._core import _accelerator
from cupy.cuda import cub
from cupy.cuda import common
from cupy.cuda import runtime
def _get_bin_edges(a, bins, range):
    """
    Computes the bins used internally by `histogram`.

    Args:
        a (ndarray): Ravelled data array
        bins (int or ndarray): Forwarded argument from `histogram`.
        range (None or tuple): Forwarded argument from `histogram`.

    Returns:
        bin_edges (ndarray): Array of bin edges
    """
    n_equal_bins = None
    bin_edges = None
    if isinstance(bins, str):
        raise NotImplementedError('only integer and array bins are implemented')
    elif isinstance(bins, cupy.ndarray) or numpy.ndim(bins) == 1:
        if isinstance(bins, cupy.ndarray):
            bin_edges = bins
        else:
            bin_edges = numpy.asarray(bins)
        if (bin_edges[:-1] > bin_edges[1:]).any():
            raise ValueError('`bins` must increase monotonically, when an array')
        if isinstance(bin_edges, numpy.ndarray):
            bin_edges = cupy.asarray(bin_edges)
    elif numpy.ndim(bins) == 0:
        try:
            n_equal_bins = operator.index(bins)
        except TypeError:
            raise TypeError('`bins` must be an integer, a string, or an array')
        if n_equal_bins < 1:
            raise ValueError('`bins` must be positive, when an integer')
        first_edge, last_edge = _get_outer_edges(a, range)
    else:
        raise ValueError('`bins` must be 1d, when an array')
    if n_equal_bins is not None:
        bin_type = cupy.result_type(first_edge, last_edge, a)
        if cupy.issubdtype(bin_type, cupy.integer):
            bin_type = cupy.result_type(bin_type, float)
        bin_edges = cupy.linspace(first_edge, last_edge, n_equal_bins + 1, endpoint=True, dtype=bin_type)
    return bin_edges