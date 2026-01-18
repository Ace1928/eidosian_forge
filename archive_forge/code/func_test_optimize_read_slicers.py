import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def test_optimize_read_slicers():
    assert optimize_read_slicers((1,), (10,), 4, _never) == ((1,), ())
    assert optimize_read_slicers((slice(None),), (10,), 4, _never) == ((slice(None),), (slice(None),))
    assert optimize_read_slicers((slice(9),), (10,), 4, _never) == ((slice(0, 9, 1),), (slice(None),))
    assert optimize_read_slicers((slice(9),), (10,), 4, _always) == ((slice(0, 9, 1),), (slice(None),))
    assert optimize_read_slicers((slice(0, 9, 2),), (10,), 4, _never) == ((slice(0, 9, 2),), (slice(None),))
    assert optimize_read_slicers((slice(0, 9, 2),), (10,), 4, _always) == ((slice(0, 9, 1),), (slice(None, None, 2),))
    assert optimize_read_slicers((1,), (10,), 4, _always) == ((1,), ())
    assert optimize_read_slicers((slice(None), slice(None)), (10, 6), 4, _never) == ((slice(None), slice(None)), (slice(None), slice(None)))
    assert optimize_read_slicers((slice(None), 1), (10, 6), 4, _never) == ((slice(None), 1), (slice(None),))
    assert optimize_read_slicers((1, slice(None)), (10, 6), 4, _never) == ((1, slice(None)), (slice(None),))
    assert optimize_read_slicers((slice(9), slice(None)), (10, 6), 4, _never) == ((slice(0, 9, 1), slice(None)), (slice(None), slice(None)))
    assert optimize_read_slicers((slice(9), slice(None)), (10, 6), 4, _always) == ((slice(None), slice(None)), (slice(0, 9, 1), slice(None)))
    assert optimize_read_slicers((slice(None), slice(5)), (10, 6), 4, _always) == ((slice(None), slice(0, 5, 1)), (slice(None), slice(None)))
    assert optimize_read_slicers((slice(0, 9, 3), slice(None)), (10, 6), 4, _never) == ((slice(0, 9, 3), slice(None)), (slice(None), slice(None)))
    assert optimize_read_slicers((slice(0, 9, 3), slice(None)), (10, 6), 4, _always) == ((slice(None), slice(None)), (slice(0, 9, 3), slice(None)))
    assert optimize_read_slicers((slice(0, 9, 3), slice(None)), (10, 6), 4, _partial) == ((slice(0, 9, 1), slice(None)), (slice(None, None, 3), slice(None)))
    assert optimize_read_slicers((slice(None), slice(0, 5, 2)), (10, 6), 4, _never) == ((slice(None), slice(0, 5, 2)), (slice(None), slice(None)))
    assert optimize_read_slicers((slice(None), slice(0, 5, 2)), (10, 6), 4, _always) == ((slice(None), slice(0, 5, 1)), (slice(None), slice(None, None, 2)))
    assert optimize_read_slicers((slice(None), 1), (10, 6), 4, _always) == ((slice(None), 1), (slice(None),))
    _depends0 = partial(threshold_heuristic, skip_thresh=10 * 4 - 1)
    _depends1 = partial(threshold_heuristic, skip_thresh=10 * 4)
    assert optimize_read_slicers((slice(9), slice(None), slice(None)), (10, 6, 2), 4, _depends0) == ((slice(None), slice(None), slice(None)), (slice(0, 9, 1), slice(None), slice(None)))
    assert optimize_read_slicers((slice(None), slice(5), slice(None)), (10, 6, 2), 4, _depends0) == ((slice(None), slice(0, 5, 1), slice(None)), (slice(None), slice(None), slice(None)))
    assert optimize_read_slicers((slice(None), slice(5), slice(None)), (10, 6, 2), 4, _depends1) == ((slice(None), slice(None), slice(None)), (slice(None), slice(0, 5, 1), slice(None)))
    sn = slice(None)
    assert optimize_read_slicers((1, 2, 3), (2, 3, 4), 4, _always) == ((sn, sn, 3), (1, 2))