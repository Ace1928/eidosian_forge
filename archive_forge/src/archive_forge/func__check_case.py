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
def _check_case(ty):
    arr = pa.array(['string', np.nan], type=ty, from_pandas=True)
    expected = pa.array(['string', None], type=ty)
    assert arr.equals(expected)