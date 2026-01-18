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
def test_sum_forwarding_implementation(self):

    class MyArray(np.ndarray):

        def sum(self, axis, out):
            return 'summed'

        def __array_function__(self, func, types, args, kwargs):
            return super().__array_function__(func, types, args, kwargs)
    array = np.array(1).view(MyArray)
    assert_equal(np.sum(array), 'summed')