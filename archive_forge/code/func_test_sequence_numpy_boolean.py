import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
@parametrize_with_sequence_types
def test_sequence_numpy_boolean(seq):
    expected = [np.bool_(True), None, np.bool_(False), None]
    arr = pa.array(seq(expected))
    assert arr.type == pa.bool_()
    assert arr.to_pylist() == [True, None, False, None]