from collections import abc
import email
from email.parser import Parser
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('df,kwargs,expected', [(DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=list('abc')).set_index(['a', 'b']), {'column_dtypes': 'float64', 'index_dtypes': {0: 'int32', 1: 'int8'}}, np.rec.array([(1, 2, 3.0), (4, 5, 6.0), (7, 8, 9.0)], dtype=[('a', f'{tm.ENDIAN}i4'), ('b', 'i1'), ('c', f'{tm.ENDIAN}f8')])), (DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=MultiIndex.from_tuples([('a', 'd'), ('b', 'e'), ('c', 'f')])), {'column_dtypes': {0: f'{tm.ENDIAN}U1', 2: 'float32'}, 'index_dtypes': 'float32'}, np.rec.array([(0.0, '1', 2, 3.0), (1.0, '4', 5, 6.0), (2.0, '7', 8, 9.0)], dtype=[('index', f'{tm.ENDIAN}f4'), ("('a', 'd')", f'{tm.ENDIAN}U1'), ("('b', 'e')", f'{tm.ENDIAN}i8'), ("('c', 'f')", f'{tm.ENDIAN}f4')])), (DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=MultiIndex.from_tuples([('a', 'd'), ('b', 'e'), ('c', 'f')], names=list('ab')), index=MultiIndex.from_tuples([('d', -4), ('d', -5), ('f', -6)], names=list('cd'))), {'column_dtypes': 'float64', 'index_dtypes': {0: f'{tm.ENDIAN}U2', 1: 'int8'}}, np.rec.array([('d', -4, 1.0, 2.0, 3.0), ('d', -5, 4.0, 5.0, 6.0), ('f', -6, 7, 8, 9.0)], dtype=[('c', f'{tm.ENDIAN}U2'), ('d', 'i1'), ("('a', 'd')", f'{tm.ENDIAN}f8'), ("('b', 'e')", f'{tm.ENDIAN}f8'), ("('c', 'f')", f'{tm.ENDIAN}f8')]))])
def test_to_records_dtype_mi(self, df, kwargs, expected):
    result = df.to_records(**kwargs)
    tm.assert_almost_equal(result, expected)