from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
def test_to_numpy_dictionary():
    arr = pa.array(['a', 'b', 'a']).dictionary_encode()
    expected = np.array(['a', 'b', 'a'], dtype=object)
    np_arr = arr.to_numpy(zero_copy_only=False)
    np.testing.assert_array_equal(np_arr, expected)