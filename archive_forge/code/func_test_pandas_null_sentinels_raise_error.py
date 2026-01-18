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
def test_pandas_null_sentinels_raise_error():
    cases = [([None, np.nan], 'null'), (['string', np.nan], 'binary'), (['string', np.nan], 'utf8'), (['string', np.nan], 'large_binary'), (['string', np.nan], 'large_utf8'), ([b'string', np.nan], pa.binary(6)), ([True, np.nan], pa.bool_()), ([decimal.Decimal('0'), np.nan], pa.decimal128(12, 2)), ([0, np.nan], pa.date32()), ([0, np.nan], pa.date32()), ([0, np.nan], pa.date64()), ([0, np.nan], pa.time32('s')), ([0, np.nan], pa.time64('us')), ([0, np.nan], pa.timestamp('us')), ([0, np.nan], pa.duration('us'))]
    for case, ty in cases:
        with pytest.raises((ValueError, TypeError)):
            pa.array(case, type=ty)
        result = pa.array(case, type=ty, from_pandas=True)
        assert result.null_count == (1 if ty != 'null' else 2)