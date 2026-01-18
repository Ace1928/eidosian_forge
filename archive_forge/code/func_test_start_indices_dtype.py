from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
def test_start_indices_dtype():
    rarray = RaggedArray([[]], dtype='int64')
    assert rarray.start_indices.dtype == np.dtype('uint8')
    np.testing.assert_array_equal(rarray.start_indices, [0])
    rarray = RaggedArray([[23, 24]], dtype='int64')
    assert rarray.start_indices.dtype == np.dtype('uint8')
    np.testing.assert_array_equal(rarray.start_indices, [0])
    max_uint8 = np.iinfo('uint8').max
    rarray = RaggedArray([np.zeros(max_uint8), []], dtype='int64')
    assert rarray.start_indices.dtype == np.dtype('uint8')
    np.testing.assert_array_equal(rarray.start_indices, [0, max_uint8])
    rarray = RaggedArray([np.zeros(max_uint8 + 1), []], dtype='int64')
    assert rarray.start_indices.dtype == np.dtype('uint16')
    np.testing.assert_array_equal(rarray.start_indices, [0, max_uint8 + 1])
    max_uint16 = np.iinfo('uint16').max
    rarray = RaggedArray([np.zeros(max_uint16), []], dtype='int64')
    assert rarray.start_indices.dtype == np.dtype('uint16')
    np.testing.assert_array_equal(rarray.start_indices, [0, max_uint16])
    rarray = RaggedArray([np.zeros(max_uint16 + 1), []], dtype='int64')
    assert rarray.start_indices.dtype == np.dtype('uint32')
    np.testing.assert_array_equal(rarray.start_indices, [0, max_uint16 + 1])