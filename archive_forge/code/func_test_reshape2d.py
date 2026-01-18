from __future__ import annotations
import itertools
import pickle
from typing import Any
from unittest.mock import patch, Mock
from datetime import datetime, date, timedelta
import numpy as np
from numpy.testing import (assert_array_equal, assert_approx_equal,
import pytest
from matplotlib import _api, cbook
import matplotlib.colors as mcolors
from matplotlib.cbook import delete_masked_points, strip_math
def test_reshape2d():

    class Dummy:
        pass
    xnew = cbook._reshape_2D([], 'x')
    assert np.shape(xnew) == (1, 0)
    x = [Dummy() for _ in range(5)]
    xnew = cbook._reshape_2D(x, 'x')
    assert np.shape(xnew) == (1, 5)
    x = np.arange(5)
    xnew = cbook._reshape_2D(x, 'x')
    assert np.shape(xnew) == (1, 5)
    x = [[Dummy() for _ in range(5)] for _ in range(3)]
    xnew = cbook._reshape_2D(x, 'x')
    assert np.shape(xnew) == (3, 5)
    x = np.random.rand(3, 5)
    xnew = cbook._reshape_2D(x, 'x')
    assert np.shape(xnew) == (5, 3)
    x = [[1], [2], [3]]
    xnew = cbook._reshape_2D(x, 'x')
    assert isinstance(xnew, list)
    assert isinstance(xnew[0], np.ndarray) and xnew[0].shape == (1,)
    assert isinstance(xnew[1], np.ndarray) and xnew[1].shape == (1,)
    assert isinstance(xnew[2], np.ndarray) and xnew[2].shape == (1,)
    x = [np.array(0), np.array(1), np.array(2)]
    xnew = cbook._reshape_2D(x, 'x')
    assert isinstance(xnew, list)
    assert len(xnew) == 1
    assert isinstance(xnew[0], np.ndarray) and xnew[0].shape == (3,)
    x = [[1, 2, 3], [3, 4], [2]]
    xnew = cbook._reshape_2D(x, 'x')
    assert isinstance(xnew, list)
    assert isinstance(xnew[0], np.ndarray) and xnew[0].shape == (3,)
    assert isinstance(xnew[1], np.ndarray) and xnew[1].shape == (2,)
    assert isinstance(xnew[2], np.ndarray) and xnew[2].shape == (1,)

    class ArraySubclass(np.ndarray):

        def __iter__(self):
            for value in super().__iter__():
                yield np.array(value)

        def __getitem__(self, item):
            return np.array(super().__getitem__(item))
    v = np.arange(10, dtype=float)
    x = ArraySubclass((10,), dtype=float, buffer=v.data)
    xnew = cbook._reshape_2D(x, 'x')
    assert len(xnew) == 1
    assert isinstance(xnew[0], ArraySubclass)
    x = ['a', 'b', 'c', 'c', 'dd', 'e', 'f', 'ff', 'f']
    xnew = cbook._reshape_2D(x, 'x')
    assert len(xnew[0]) == len(x)
    assert isinstance(xnew[0], np.ndarray)