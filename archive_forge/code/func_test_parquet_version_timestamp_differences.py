import datetime
import io
import warnings
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.pandas
def test_parquet_version_timestamp_differences():
    i_s = pd.Timestamp('2010-01-01').value / 1000000000
    d_s = np.arange(i_s, i_s + 10, 1, dtype='int64')
    d_ms = d_s * 1000
    d_us = d_ms * 1000
    d_ns = d_us * 1000
    a_s = pa.array(d_s, type=pa.timestamp('s'))
    a_ms = pa.array(d_ms, type=pa.timestamp('ms'))
    a_us = pa.array(d_us, type=pa.timestamp('us'))
    a_ns = pa.array(d_ns, type=pa.timestamp('ns'))
    all_versions = ['1.0', '2.4', '2.6']
    names = ['ts:s', 'ts:ms', 'ts:us', 'ts:ns']
    table = pa.Table.from_arrays([a_s, a_ms, a_us, a_ns], names)
    expected = pa.Table.from_arrays([a_ms, a_ms, a_us, a_us], names)
    _check_roundtrip(table, expected, version='1.0')
    _check_roundtrip(table, expected, version='2.4')
    expected = pa.Table.from_arrays([a_ms, a_ms, a_us, a_ns], names)
    _check_roundtrip(table, expected, version='2.6')
    expected = pa.Table.from_arrays([a_ms, a_ms, a_ms, a_ms], names)
    for ver in all_versions:
        _check_roundtrip(table, expected, coerce_timestamps='ms', version=ver)
    expected = pa.Table.from_arrays([a_us, a_us, a_us, a_us], names)
    for ver in all_versions:
        _check_roundtrip(table, expected, version=ver, coerce_timestamps='us')
    expected = pa.Table.from_arrays([a_ns, a_ns, a_ns, a_ns], names)
    for ver in all_versions:
        _check_roundtrip(table, expected, version=ver, use_deprecated_int96_timestamps=True)