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
@pytest.mark.large_memory
@pytest.mark.parametrize('ty', [pa.binary(), pa.string()])
def test_string_too_large(ty):
    s = b'0123456789abcdefghijklmnopqrstuvwxyz'
    nrepeats = math.ceil((2 ** 32 + 5) / len(s))
    with pytest.raises(pa.ArrowCapacityError):
        pa.array([b'foo', s * nrepeats, None, b'bar'], type=ty)