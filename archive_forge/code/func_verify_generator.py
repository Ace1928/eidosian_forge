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
def verify_generator(generator, buckets, probs, nsamples=1000000, nrepeat=5, success_rate=0.2, alpha=0.05):
    """Verify whether the generator is correct using chi-square testing.

    The test is repeated for "nrepeat" times and we check if the success rate is
     above the threshold (25% by default).

    Parameters
    ----------
    generator: function
        A function that is assumed to generate i.i.d samples from a specific distribution.
            generator(N) should generate N random samples.
    buckets: list of tuple or list of number
        The buckets to run the chi-square the test. Make sure that the buckets cover
         the whole range of the distribution. Also, the buckets must be in ascending order and
         have no intersection
    probs: list or tuple
        The ground-truth probability of the random value fall in a specific bucket.
    nsamples: int
        The number of samples to generate for the testing
    nrepeat: int
        The times to repeat the test
    success_rate: float
        The desired success rate
    alpha: float
        The desired threshold for type-I error i.e. when a true null hypothesis is rejected

    Returns
    -------
    cs_ret_l: list
        The p values of the chi-square test.
    """
    cs_ret_l = []
    obs_freq_l = []
    expected_freq_l = []
    for _ in range(nrepeat):
        cs_ret, obs_freq, expected_freq = chi_square_check(generator=generator, buckets=buckets, probs=probs, nsamples=nsamples)
        cs_ret_l.append(cs_ret)
        obs_freq_l.append(obs_freq)
        expected_freq_l.append(expected_freq)
    success_num = (np.array(cs_ret_l) > alpha).sum()
    if success_num < nrepeat * success_rate:
        raise AssertionError('Generator test fails, Chi-square p=%s, obs_freq=%s, expected_freq=%s.\nbuckets=%s, probs=%s' % (str(cs_ret_l), str(obs_freq_l), str(expected_freq_l), str(buckets), str(probs)))
    return cs_ret_l