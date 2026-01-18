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
def test_sequence_decimal_infer_mixed():
    cases = [([decimal.Decimal('1.234'), decimal.Decimal('3.456')], pa.decimal128(4, 3)), ([decimal.Decimal('1.234'), decimal.Decimal('456.7')], pa.decimal128(6, 3)), ([decimal.Decimal('123.4'), decimal.Decimal('4.567')], pa.decimal128(6, 3)), ([decimal.Decimal('123e2'), decimal.Decimal('4567e3')], pa.decimal128(7, 0)), ([decimal.Decimal('123e4'), decimal.Decimal('4567e2')], pa.decimal128(7, 0)), ([decimal.Decimal('0.123'), decimal.Decimal('0.04567')], pa.decimal128(5, 5)), ([decimal.Decimal('0.001'), decimal.Decimal('1.01E5')], pa.decimal128(9, 3))]
    for data, typ in cases:
        assert pa.infer_type(data) == typ
        arr = pa.array(data)
        assert arr.type == typ
        assert arr.to_pylist() == data