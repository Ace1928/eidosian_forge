import sys
import __future__
import six
import numpy as np
import pytest
from patsy import PatsyError
from patsy.design_info import DesignMatrix, DesignInfo
from patsy.eval import EvalEnvironment
from patsy.desc import ModelDesc, Term, INTERCEPT
from patsy.categorical import C
from patsy.contrasts import Helmert
from patsy.user_util import balanced, LookupFactor
from patsy.build import (design_matrix_builders,
from patsy.highlevel import *
from patsy.util import (have_pandas,
from patsy.origin import Origin
def test_return_pandas():
    if not have_pandas:
        return
    s1 = pandas.Series([1, 2, 3], name='AA', index=[10, 20, 30])
    s2 = pandas.Series([4, 5, 6], name='BB', index=[10, 20, 30])
    df1 = dmatrix('s1', return_type='dataframe')
    assert np.allclose(df1, [[1, 1], [1, 2], [1, 3]])
    assert np.array_equal(df1.columns, ['Intercept', 's1'])
    assert df1.design_info.column_names == ['Intercept', 's1']
    assert np.array_equal(df1.index, [10, 20, 30])
    df2, df3 = dmatrices('s2 ~ s1', return_type='dataframe')
    assert np.allclose(df2, [[4], [5], [6]])
    assert np.array_equal(df2.columns, ['s2'])
    assert df2.design_info.column_names == ['s2']
    assert np.array_equal(df2.index, [10, 20, 30])
    assert np.allclose(df3, [[1, 1], [1, 2], [1, 3]])
    assert np.array_equal(df3.columns, ['Intercept', 's1'])
    assert df3.design_info.column_names == ['Intercept', 's1']
    assert np.array_equal(df3.index, [10, 20, 30])
    df4 = dmatrix(s1, return_type='dataframe')
    assert np.allclose(df4, [[1], [2], [3]])
    assert np.array_equal(df4.columns, ['AA'])
    assert df4.design_info.column_names == ['AA']
    assert np.array_equal(df4.index, [10, 20, 30])
    df5, df6 = dmatrices((s2, s1), return_type='dataframe')
    assert np.allclose(df5, [[4], [5], [6]])
    assert np.array_equal(df5.columns, ['BB'])
    assert df5.design_info.column_names == ['BB']
    assert np.array_equal(df5.index, [10, 20, 30])
    assert np.allclose(df6, [[1], [2], [3]])
    assert np.array_equal(df6.columns, ['AA'])
    assert df6.design_info.column_names == ['AA']
    assert np.array_equal(df6.index, [10, 20, 30])
    df7, df8 = dmatrices((s1, [10, 11, 12]), return_type='dataframe')
    assert np.array_equal(df7.index, s1.index)
    assert np.array_equal(df8.index, s1.index)
    df9, df10 = dmatrices(([10, 11, 12], s1), return_type='dataframe')
    assert np.array_equal(df9.index, s1.index)
    assert np.array_equal(df10.index, s1.index)
    import patsy.highlevel
    had_pandas = patsy.highlevel.have_pandas
    try:
        patsy.highlevel.have_pandas = False
        pytest.raises(PatsyError, dmatrix, 'x', {'x': [1]}, 0, return_type='dataframe')
        pytest.raises(PatsyError, dmatrices, 'y ~ x', {'x': [1], 'y': [2]}, 0, return_type='dataframe')
    finally:
        patsy.highlevel.have_pandas = had_pandas