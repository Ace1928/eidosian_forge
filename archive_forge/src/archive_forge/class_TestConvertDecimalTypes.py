import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
class TestConvertDecimalTypes:
    """
    Conversion test for decimal types.
    """
    decimal32 = [decimal.Decimal('-1234.123'), decimal.Decimal('1234.439')]
    decimal64 = [decimal.Decimal('-129934.123331'), decimal.Decimal('129534.123731')]
    decimal128 = [decimal.Decimal('394092382910493.12341234678'), decimal.Decimal('-314292388910493.12343437128')]

    @pytest.mark.parametrize(('values', 'expected_type'), [pytest.param(decimal32, pa.decimal128(7, 3), id='decimal32'), pytest.param(decimal64, pa.decimal128(12, 6), id='decimal64'), pytest.param(decimal128, pa.decimal128(26, 11), id='decimal128')])
    def test_decimal_from_pandas(self, values, expected_type):
        expected = pd.DataFrame({'decimals': values})
        table = pa.Table.from_pandas(expected, preserve_index=False)
        field = pa.field('decimals', expected_type)
        expected_schema = pa.schema([field], metadata=table.schema.metadata)
        assert table.schema.equals(expected_schema)

    @pytest.mark.parametrize('values', [pytest.param(decimal32, id='decimal32'), pytest.param(decimal64, id='decimal64'), pytest.param(decimal128, id='decimal128')])
    def test_decimal_to_pandas(self, values):
        expected = pd.DataFrame({'decimals': values})
        converted = pa.Table.from_pandas(expected)
        df = converted.to_pandas()
        tm.assert_frame_equal(df, expected)

    def test_decimal_fails_with_truncation(self):
        data1 = [decimal.Decimal('1.234')]
        type1 = pa.decimal128(10, 2)
        with pytest.raises(pa.ArrowInvalid):
            pa.array(data1, type=type1)
        data2 = [decimal.Decimal('1.2345')]
        type2 = pa.decimal128(10, 3)
        with pytest.raises(pa.ArrowInvalid):
            pa.array(data2, type=type2)

    def test_decimal_with_different_precisions(self):
        data = [decimal.Decimal('0.01'), decimal.Decimal('0.001')]
        series = pd.Series(data)
        array = pa.array(series)
        assert array.to_pylist() == data
        assert array.type == pa.decimal128(3, 3)
        array = pa.array(data, type=pa.decimal128(12, 5))
        expected = [decimal.Decimal('0.01000'), decimal.Decimal('0.00100')]
        assert array.to_pylist() == expected

    def test_decimal_with_None_explicit_type(self):
        series = pd.Series([decimal.Decimal('3.14'), None])
        _check_series_roundtrip(series, type_=pa.decimal128(12, 5))
        series = pd.Series([None] * 2)
        _check_series_roundtrip(series, type_=pa.decimal128(12, 5))

    def test_decimal_with_None_infer_type(self):
        series = pd.Series([decimal.Decimal('3.14'), None])
        _check_series_roundtrip(series, expected_pa_type=pa.decimal128(3, 2))

    def test_strided_objects(self, tmpdir):
        data = {'a': {0: 'a'}, 'b': {0: decimal.Decimal('0.0')}}
        df = pd.DataFrame.from_dict(data)
        _check_pandas_roundtrip(df)