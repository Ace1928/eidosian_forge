import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def _gen_gaussians_even(sigmas, total_length):
    num_peaks = len(sigmas)
    delta = total_length / (num_peaks + 1)
    center_locs = np.linspace(delta, total_length - delta, num=num_peaks).astype(int)
    out_data = _gen_gaussians(center_locs, sigmas, total_length)
    return (out_data, center_locs)