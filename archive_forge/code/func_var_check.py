import time
import gzip
import struct
import traceback
import numbers
import sys
import os
import platform
import errno
import logging
import bz2
import zipfile
import json
from contextlib import contextmanager
from collections import OrderedDict
import numpy as np
import numpy.testing as npt
import numpy.random as rnd
import mxnet as mx
from .context import Context, current_context
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID
from .ndarray import array
from .symbol import Symbol
from .symbol.numpy import _Symbol as np_symbol
from .util import use_np, getenv, setenv  # pylint: disable=unused-import
from .runtime import Features
from .numpy_extension import get_cuda_compute_capability
def var_check(generator, sigma, nsamples=1000000):
    """Test the generator by matching the variance.
    It will need a large number of samples and is not recommended to use

    We test the sample variance by checking if it falls inside the range
        (sigma^2 - 3 * sqrt(2 * sigma^4 / (n-1)), sigma^2 + 3 * sqrt(2 * sigma^4 / (n-1)))

    References::

        @incollection{goucher2009beautiful,
              title={Beautiful Testing: Leading Professionals Reveal How They Improve Software},
              author={Goucher, Adam and Riley, Tim},
              year={2009},
              chapter=10
        }

    Examples::

        generator = lambda x: np.random.normal(0, 1.0, size=x)
        var_check_ret = var_check(generator, 0, 1.0)

    Parameters
    ----------
    generator : function
        The generator function. It's expected to generate N i.i.d samples by calling generator(N).
    sigma : float
    nsamples : int

    Returns
    -------
    ret : bool
        Whether the variance test succeeds
    """
    samples = np.array(generator(nsamples))
    sample_var = samples.var(ddof=1)
    ret = sample_var > sigma ** 2 - 3 * np.sqrt(2 * sigma ** 4 / (nsamples - 1)) and sample_var < sigma ** 2 + 3 * np.sqrt(2 * sigma ** 4 / (nsamples - 1))
    return ret