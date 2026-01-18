from scipy import fft
import numpy as np
import pytest
from numpy.testing import assert_allclose
import multiprocessing
import os
def test_set_workers_invalid():
    with pytest.raises(ValueError, match='workers must not be zero'):
        with fft.set_workers(0):
            pass
    with pytest.raises(ValueError, match='workers value out of range'):
        with fft.set_workers(-os.cpu_count() - 1):
            pass