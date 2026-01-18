import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
def test_set_module(self):
    assert_equal(np.sum.__module__, 'numpy')
    assert_equal(np.char.equal.__module__, 'numpy.char')
    assert_equal(np.fft.fft.__module__, 'numpy.fft')
    assert_equal(np.linalg.solve.__module__, 'numpy.linalg')