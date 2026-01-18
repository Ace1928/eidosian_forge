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
def test_struct_array_from_chunked():
    chunked_arr = pa.chunked_array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(TypeError, match='Expected Array'):
        pa.StructArray.from_arrays([chunked_arr], ['foo'])