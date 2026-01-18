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
def test_sequence_decimal_no_scale():
    data = [decimal.Decimal('1234234983'), decimal.Decimal('8094324')]
    for type in [pa.decimal128, pa.decimal256]:
        arr = pa.array(data, type=type(precision=10))
        assert arr.to_pylist() == data