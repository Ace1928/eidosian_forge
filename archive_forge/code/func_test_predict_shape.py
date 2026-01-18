import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def test_predict_shape():
    shapes = (15, 16, 17, 18)
    for n_dim in range(len(shapes)):
        shape = shapes[:n_dim + 1]
        arr = np.arange(np.prod(shape)).reshape(shape)
        slicers_list = []
        for i in range(n_dim):
            slicers_list.append(_slices_for_len(shape[i]))
            for sliceobj in product(*slicers_list):
                assert predict_shape(sliceobj, shape) == arr[sliceobj].shape
    assert predict_shape((Ellipsis,), (2, 3)) == (2, 3)
    assert predict_shape((Ellipsis, 1), (2, 3)) == (2,)
    assert predict_shape((1, Ellipsis), (2, 3)) == (3,)
    assert predict_shape((1, slice(None), Ellipsis), (2, 3)) == (3,)
    assert predict_shape((None,), (2, 3)) == (1, 2, 3)
    assert predict_shape((None, 1), (2, 3)) == (1, 3)
    assert predict_shape((1, None, slice(None)), (2, 3)) == (1, 3)
    assert predict_shape((1, slice(None), None), (2, 3)) == (3, 1)