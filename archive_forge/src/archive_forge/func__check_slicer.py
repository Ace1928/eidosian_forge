import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def _check_slicer(sliceobj, arr, fobj, offset, order, heuristic=threshold_heuristic):
    new_slice = fileslice(fobj, sliceobj, arr.shape, arr.dtype, offset, order, heuristic)
    assert_array_equal(arr[sliceobj], new_slice)