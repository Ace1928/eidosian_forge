from scipy import fft
import numpy as np
import pytest
from numpy.testing import assert_allclose
import multiprocessing
import os
def test_set_get_workers():
    cpus = os.cpu_count()
    assert fft.get_workers() == 1
    with fft.set_workers(4):
        assert fft.get_workers() == 4
        with fft.set_workers(-1):
            assert fft.get_workers() == cpus
        assert fft.get_workers() == 4
    assert fft.get_workers() == 1
    with fft.set_workers(-cpus):
        assert fft.get_workers() == 1