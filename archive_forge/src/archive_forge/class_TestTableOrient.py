from collections import OrderedDict
import datetime as dt
import decimal
from io import StringIO
import json
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.string_ import StringDtype
from pandas.core.series import Series
from pandas.tests.extension.date import (
from pandas.tests.extension.decimal.array import (
from pandas.io.json._table_schema import (
class TestTableOrient:

    @pytest.fixture
    def da(self):
        return DateArray([dt.date(2021, 10, 10)])

    @pytest.fixture
    def dc(self):
        return DecimalArray([decimal.Decimal(10)])

    @pytest.fixture
    def sa(self):
        return array(['pandas'], dtype='string')

    @pytest.fixture
    def ia(self):
        return array([10], dtype='Int64')

    @pytest.fixture
    def df(self, da, dc, sa, ia):
        return DataFrame({'A': da, 'B': dc, 'C': sa, 'D': ia})

    def test_build_date_series(self, da):
        s = Series(da, name='a')
        s.index.name = 'id'
        result = s.to_json(orient='table', date_format='iso')
        result = json.loads(result, object_pairs_hook=OrderedDict)
        assert 'pandas_version' in result['schema']
        result['schema'].pop('pandas_version')
        fields = [{'name': 'id', 'type': 'integer'}, {'name': 'a', 'type': 'any', 'extDtype': 'DateDtype'}]
        schema = {'fields': fields, 'primaryKey': ['id']}
        expected = OrderedDict([('schema', schema), ('data', [OrderedDict([('id', 0), ('a', '2021-10-10T00:00:00.000')])])])
        assert result == expected

    def test_build_decimal_series(self, dc):
        s = Series(dc, name='a')
        s.index.name = 'id'
        result = s.to_json(orient='table', date_format='iso')
        result = json.loads(result, object_pairs_hook=OrderedDict)
        assert 'pandas_version' in result['schema']
        result['schema'].pop('pandas_version')
        fields = [{'name': 'id', 'type': 'integer'}, {'name': 'a', 'type': 'number', 'extDtype': 'decimal'}]
        schema = {'fields': fields, 'primaryKey': ['id']}
        expected = OrderedDict([('schema', schema), ('data', [OrderedDict([('id', 0), ('a', 10.0)])])])
        assert result == expected

    def test_build_string_series(self, sa):
        s = Series(sa, name='a')
        s.index.name = 'id'
        result = s.to_json(orient='table', date_format='iso')
        result = json.loads(result, object_pairs_hook=OrderedDict)
        assert 'pandas_version' in result['schema']
        result['schema'].pop('pandas_version')
        fields = [{'name': 'id', 'type': 'integer'}, {'name': 'a', 'type': 'any', 'extDtype': 'string'}]
        schema = {'fields': fields, 'primaryKey': ['id']}
        expected = OrderedDict([('schema', schema), ('data', [OrderedDict([('id', 0), ('a', 'pandas')])])])
        assert result == expected

    def test_build_int64_series(self, ia):
        s = Series(ia, name='a')
        s.index.name = 'id'
        result = s.to_json(orient='table', date_format='iso')
        result = json.loads(result, object_pairs_hook=OrderedDict)
        assert 'pandas_version' in result['schema']
        result['schema'].pop('pandas_version')
        fields = [{'name': 'id', 'type': 'integer'}, {'name': 'a', 'type': 'integer', 'extDtype': 'Int64'}]
        schema = {'fields': fields, 'primaryKey': ['id']}
        expected = OrderedDict([('schema', schema), ('data', [OrderedDict([('id', 0), ('a', 10)])])])
        assert result == expected

    def test_to_json(self, df):
        df = df.copy()
        df.index.name = 'idx'
        result = df.to_json(orient='table', date_format='iso')
        result = json.loads(result, object_pairs_hook=OrderedDict)
        assert 'pandas_version' in result['schema']
        result['schema'].pop('pandas_version')
        fields = [OrderedDict({'name': 'idx', 'type': 'integer'}), OrderedDict({'name': 'A', 'type': 'any', 'extDtype': 'DateDtype'}), OrderedDict({'name': 'B', 'type': 'number', 'extDtype': 'decimal'}), OrderedDict({'name': 'C', 'type': 'any', 'extDtype': 'string'}), OrderedDict({'name': 'D', 'type': 'integer', 'extDtype': 'Int64'})]
        schema = OrderedDict({'fields': fields, 'primaryKey': ['idx']})
        data = [OrderedDict([('idx', 0), ('A', '2021-10-10T00:00:00.000'), ('B', 10.0), ('C', 'pandas'), ('D', 10)])]
        expected = OrderedDict([('schema', schema), ('data', data)])
        assert result == expected

    def test_json_ext_dtype_reading_roundtrip(self):
        df = DataFrame({'a': Series([2, NA], dtype='Int64'), 'b': Series([1.5, NA], dtype='Float64'), 'c': Series([True, NA], dtype='boolean')}, index=Index([1, NA], dtype='Int64'))
        expected = df.copy()
        data_json = df.to_json(orient='table', indent=4)
        result = read_json(StringIO(data_json), orient='table')
        tm.assert_frame_equal(result, expected)

    def test_json_ext_dtype_reading(self):
        data_json = '{\n            "schema":{\n                "fields":[\n                    {\n                        "name":"a",\n                        "type":"integer",\n                        "extDtype":"Int64"\n                    }\n                ],\n            },\n            "data":[\n                {\n                    "a":2\n                },\n                {\n                    "a":null\n                }\n            ]\n        }'
        result = read_json(StringIO(data_json), orient='table')
        expected = DataFrame({'a': Series([2, NA], dtype='Int64')})
        tm.assert_frame_equal(result, expected)