import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def test_slice2outax():
    sn = slice(None)
    assert slice2outax(1, (sn,)) == (0,)
    assert slice2outax(1, (1,)) == (None,)
    assert slice2outax(1, (None,)) == (1,)
    assert slice2outax(1, (None, 1)) == (None,)
    assert slice2outax(1, (None, 1, None)) == (None,)
    assert slice2outax(1, (None, sn)) == (1,)
    assert slice2outax(2, (sn,)) == (0, 1)
    assert slice2outax(2, (sn, sn)) == (0, 1)
    assert slice2outax(2, (1,)) == (None, 0)
    assert slice2outax(2, (sn, 1)) == (0, None)
    assert slice2outax(2, (None,)) == (1, 2)
    assert slice2outax(2, (None, 1)) == (None, 1)
    assert slice2outax(2, (None, 1, None)) == (None, 2)
    assert slice2outax(2, (None, 1, None, 2)) == (None, None)
    assert slice2outax(2, (None, sn, None, 1)) == (1, None)
    assert slice2outax(3, (sn,)) == (0, 1, 2)
    assert slice2outax(3, (sn, sn)) == (0, 1, 2)
    assert slice2outax(3, (sn, None, sn)) == (0, 2, 3)
    assert slice2outax(3, (sn, None, sn, None, sn)) == (0, 2, 4)
    assert slice2outax(3, (1,)) == (None, 0, 1)
    assert slice2outax(3, (None, sn, None, 1)) == (1, None, 3)