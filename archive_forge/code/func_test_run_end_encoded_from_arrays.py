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
def test_run_end_encoded_from_arrays():
    check_run_end_encoded_from_arrays_with_type()
    for run_end_type in [pa.int16(), pa.int32(), pa.int64()]:
        for value_type in [pa.uint32(), pa.int32(), pa.uint64(), pa.int64()]:
            ree_type = pa.run_end_encoded(run_end_type, value_type)
            check_run_end_encoded_from_arrays_with_type(ree_type)