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
@pytest.mark.pandas
def test_pandas_null_sentinels_index():
    import pandas as pd
    idx = pd.Index([1, 2, np.nan], dtype=object)
    result = pa.array(idx)
    expected = pa.array([1, 2, np.nan], from_pandas=True)
    assert result.equals(expected)