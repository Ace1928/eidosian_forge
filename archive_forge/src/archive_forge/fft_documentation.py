from __future__ import annotations
import inspect
import warnings
from collections.abc import Sequence
import numpy as np
from dask.array.core import asarray
from dask.array.core import concatenate as _concatenate
from dask.array.creation import arange as _arange
from dask.array.numpy_compat import NUMPY_GE_200
from dask.utils import derived_from, skip_doctest
Wrap 1D, 2D, and ND real and complex FFT functions

    Takes a function that behaves like ``numpy.fft`` functions and
    a specified kind to match it to that are named after the functions
    in the ``numpy.fft`` API.

    Supported kinds include:

        * fft
        * fft2
        * fftn
        * ifft
        * ifft2
        * ifftn
        * rfft
        * rfft2
        * rfftn
        * irfft
        * irfft2
        * irfftn
        * hfft
        * ihfft

    Examples
    --------
    >>> import dask.array.fft as dff
    >>> parallel_fft = dff.fft_wrap(np.fft.fft)
    >>> parallel_ifft = dff.fft_wrap(np.fft.ifft)
    