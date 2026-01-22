import os
import re
import numpy as np
import pandas
import pyarrow
import pytest
from pandas._testing import ensure_clean
from pandas.core.dtypes.common import is_list_like
from pyhdk import __version__ as hdk_version
from modin.config import StorageFormat
from modin.tests.interchange.dataframe_protocol.hdk.utils import split_df_into_chunks
from modin.tests.pandas.utils import (
from .utils import ForceHdkImport, eval_io, run_and_compare, set_execution_mode
import modin.pandas as pd
from modin.experimental.core.execution.native.implementations.hdk_on_native.calcite_serializer import (
from modin.experimental.core.execution.native.implementations.hdk_on_native.df_algebra import (
from modin.experimental.core.execution.native.implementations.hdk_on_native.partitioning.partition_manager import (
from modin.pandas.io import from_arrow
from modin.tests.pandas.utils import (
from modin.utils import try_cast_to_pandas
@pytest.mark.usefixtures('TestReadCSVFixture')
class TestCSV:
    from modin import __file__ as modin_root
    root = os.path.dirname(os.path.dirname(os.path.abspath(modin_root)) + '..')
    boston_housing_names = ['index', 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'PRICE']
    boston_housing_dtypes = {'index': 'int64', 'CRIM': 'float64', 'ZN': 'float64', 'INDUS': 'float64', 'CHAS': 'float64', 'NOX': 'float64', 'RM': 'float64', 'AGE': 'float64', 'DIS': 'float64', 'RAD': 'float64', 'TAX': 'float64', 'PTRATIO': 'float64', 'B': 'float64', 'LSTAT': 'float64', 'PRICE': 'float64'}

    def test_usecols_csv(self):
        """check with the following arguments: names, dtype, skiprows, delimiter"""
        csv_file = os.path.join(self.root, 'modin/tests/pandas/data', 'test_usecols.csv')
        for kwargs in ({'delimiter': ','}, {'sep': None}, {'skiprows': 1, 'names': ['A', 'B', 'C', 'D', 'E']}, {'dtype': {'a': 'int32', 'e': 'string'}}, {'dtype': {'a': np.dtype('int32'), 'b': np.dtype('int64'), 'e': 'string'}}):
            eval_io(fn_name='read_csv', md_extra_kwargs={'engine': 'arrow'}, filepath_or_buffer=csv_file, **kwargs)

    def test_housing_csv(self):
        csv_file = os.path.join(self.root, 'examples/data/boston_housing.csv')
        for kwargs in ({'skiprows': 1, 'names': self.boston_housing_names, 'dtype': self.boston_housing_dtypes},):
            eval_io(fn_name='read_csv', md_extra_kwargs={'engine': 'arrow'}, filepath_or_buffer=csv_file, **kwargs)

    def test_time_parsing(self):
        csv_file = os.path.join(self.root, time_parsing_csv_path)
        for kwargs in ({'skiprows': 1, 'names': ['timestamp', 'year', 'month', 'date', 'symbol', 'high', 'low', 'open', 'close', 'spread', 'volume'], 'parse_dates': ['timestamp'], 'dtype': {'symbol': 'string'}},):
            rp = pandas.read_csv(csv_file, **kwargs)
            rm = pd.read_csv(csv_file, engine='arrow', **kwargs)
            with ForceHdkImport(rm):
                rm = to_pandas(rm)
                df_equals(rm['timestamp'].dt.year, rp['timestamp'].dt.year)
                df_equals(rm['timestamp'].dt.month, rp['timestamp'].dt.month)
                df_equals(rm['timestamp'].dt.day, rp['timestamp'].dt.day)
                df_equals(rm['timestamp'].dt.hour, rp['timestamp'].dt.hour)

    def test_csv_fillna(self):
        csv_file = os.path.join(self.root, 'examples/data/boston_housing.csv')
        for kwargs in ({'skiprows': 1, 'names': self.boston_housing_names, 'dtype': self.boston_housing_dtypes},):
            eval_io(fn_name='read_csv', md_extra_kwargs={'engine': 'arrow'}, comparator=lambda df1, df2: df_equals(df1['CRIM'].fillna(1000), df2['CRIM'].fillna(1000)), filepath_or_buffer=csv_file, **kwargs)

    @pytest.mark.parametrize('null_dtype', ['category', 'float64'])
    def test_null_col(self, null_dtype):
        csv_file = os.path.join(self.root, 'modin/tests/pandas/data', 'test_null_col.csv')
        ref = pandas.read_csv(csv_file, names=['a', 'b', 'c'], dtype={'a': 'int64', 'b': 'int64', 'c': null_dtype}, skiprows=1)
        ref['a'] = ref['a'] + ref['b']
        exp = pd.read_csv(csv_file, names=['a', 'b', 'c'], dtype={'a': 'int64', 'b': 'int64', 'c': null_dtype}, skiprows=1)
        exp['a'] = exp['a'] + exp['b']
        if null_dtype == 'category':
            ref['c'] = ref['c'].astype('string')
            with ForceHdkImport(exp):
                exp = to_pandas(exp)
            exp['c'] = exp['c'].astype('string')
        df_equals(ref, exp)

    def test_read_and_concat(self):
        csv_file = os.path.join(self.root, 'modin/tests/pandas/data', 'test_usecols.csv')
        ref1 = pandas.read_csv(csv_file)
        ref2 = pandas.read_csv(csv_file)
        ref = pandas.concat([ref1, ref2])
        exp1 = pandas.read_csv(csv_file)
        exp2 = pandas.read_csv(csv_file)
        exp = pd.concat([exp1, exp2])
        with ForceHdkImport(exp):
            df_equals(ref, exp)

    @pytest.mark.parametrize('names', [None, ['a', 'b', 'c', 'd', 'e']])
    @pytest.mark.parametrize('header', [None, 0])
    def test_from_csv(self, header, names):
        csv_file = os.path.join(self.root, 'modin/tests/pandas/data', 'test_usecols.csv')
        eval_io(fn_name='read_csv', filepath_or_buffer=csv_file, header=header, names=names)

    @pytest.mark.parametrize('kwargs', [{'sep': '|'}, {'delimiter': '|'}])
    def test_sep_delimiter(self, kwargs):
        csv_file = os.path.join(self.root, 'modin/tests/pandas/data', 'test_delim.csv')
        eval_io(fn_name='read_csv', filepath_or_buffer=csv_file, **kwargs)

    @pytest.mark.skip(reason='https://github.com/modin-project/modin/issues/2174')
    def test_float32(self):
        csv_file = os.path.join(self.root, 'modin/tests/pandas/data', 'test_usecols.csv')
        kwargs = {'dtype': {'a': 'float32', 'b': 'float32'}}
        pandas_df = pandas.read_csv(csv_file, **kwargs)
        pandas_df['a'] = pandas_df['a'] + pandas_df['b']
        modin_df = pd.read_csv(csv_file, **kwargs, engine='arrow')
        modin_df['a'] = modin_df['a'] + modin_df['b']
        with ForceHdkImport(modin_df):
            df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize('engine', [None, 'arrow'])
    @pytest.mark.parametrize('parse_dates', [True, False, ['col2'], ['c2'], [['col2', 'col3']], {'col23': ['col2', 'col3']}, []])
    @pytest.mark.parametrize('names', [None, [f'c{x}' for x in range(1, 7)]])
    def test_read_csv_datetime(self, engine, parse_dates, names, request):
        parse_dates_unsupported = isinstance(parse_dates, dict) or (isinstance(parse_dates, list) and any((not isinstance(date, str) for date in parse_dates)))
        if parse_dates_unsupported and engine == 'arrow' and (not names):
            pytest.skip('In these cases Modin raises `ArrowEngineException` while pandas ' + "doesn't raise any exceptions that causes tests fails")
        skip_exc_type_check = parse_dates_unsupported and engine == 'arrow'
        if skip_exc_type_check:
            pytest.xfail(reason='https://github.com/modin-project/modin/issues/7012')
        expected_exception = None
        if 'names1-parse_dates2' in request.node.callspec.id:
            expected_exception = ValueError("Missing column provided to 'parse_dates': 'col2'")
        elif 'names1-parse_dates5-None' in request.node.callspec.id or 'names1-parse_dates4-None' in request.node.callspec.id:
            expected_exception = ValueError("Missing column provided to 'parse_dates': 'col2, col3'")
        elif 'None-parse_dates3' in request.node.callspec.id:
            expected_exception = ValueError("Missing column provided to 'parse_dates': 'c2'")
        eval_io(fn_name='read_csv', md_extra_kwargs={'engine': engine}, expected_exception=expected_exception, filepath_or_buffer=pytest.csvs_names['test_read_csv_regular'], parse_dates=parse_dates, names=names)

    @pytest.mark.parametrize('engine', [None, 'arrow'])
    @pytest.mark.parametrize('parse_dates', [None, True, False])
    def test_read_csv_datetime_tz(self, engine, parse_dates):
        with ensure_clean('.csv') as file:
            with open(file, 'w') as f:
                f.write('test\n2023-01-01T00:00:00.000-07:00')
            eval_io(fn_name='read_csv', filepath_or_buffer=file, md_extra_kwargs={'engine': engine}, parse_dates=parse_dates)

    @pytest.mark.parametrize('engine', [None, 'arrow'])
    @pytest.mark.parametrize('usecols', [None, ['col1'], ['col1', 'col1'], ['col1', 'col2', 'col6'], ['col6', 'col2', 'col1'], [0], [0, 0], [0, 1, 5], [5, 1, 0], lambda x: x in ['col1', 'col2']])
    def test_read_csv_col_handling(self, engine, usecols):
        eval_io(fn_name='read_csv', check_kwargs_callable=not callable(usecols), md_extra_kwargs={'engine': engine}, filepath_or_buffer=pytest.csvs_names['test_read_csv_regular'], usecols=usecols)

    @pytest.mark.parametrize('cols', ['c1,c2,c3', 'c1,c1,c2', 'c1,c1,c1.1,c1.2,c1', 'c1,c1,c1,c1.1,c1.2,c1.3', 'c1.1,c1.2,c1.3,c1,c1,c1', 'c1.1,c1,c1.2,c1,c1.3,c1', 'c1,c1.1,c1,c1.2,c1,c1.3', 'c1,c1,c1.1,c1.1,c1.2,c2', 'c1,c1,c1.1,c1.1,c1.2,c1.2,c2', 'c1.1,c1.1,c1,c1,c1.2,c1.2,c2', 'c1.1,c1,c1.1,c1,c1.1,c1.2,c1.2,c2'])
    def test_read_csv_duplicate_cols(self, cols):

        def test(df, lib, **kwargs):
            data = f'{cols}\n'
            with ensure_clean('.csv') as fname:
                with open(fname, 'w') as f:
                    f.write(data)
                return lib.read_csv(fname)
        run_and_compare(test, data={})

    def test_read_csv_dtype_object(self):
        with pytest.warns(UserWarning) as warns:
            with ensure_clean('.csv') as file:
                with open(file, 'w') as f:
                    f.write('test\ntest')

                def test(**kwargs):
                    return pd.read_csv(file, dtype={'test': 'object'})
                run_and_compare(test, data={})
            for warn in warns.list:
                assert not re.match('.*defaulting to pandas.*', str(warn))