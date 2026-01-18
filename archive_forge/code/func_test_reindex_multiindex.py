import os
import sys
import matplotlib
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.pandas.indexing import is_range_like
from modin.pandas.testing import assert_index_equal
from modin.tests.pandas.utils import (
from modin.utils import get_current_execution
def test_reindex_multiindex():
    data1, data2 = (np.random.randint(1, 20, (5, 5)), np.random.randint(10, 25, 6))
    index = np.array(['AUD', 'BRL', 'CAD', 'EUR', 'INR'])
    modin_midx = pd.MultiIndex.from_product([['Bank_1', 'Bank_2'], ['AUD', 'CAD', 'EUR']], names=['Bank', 'Curency'])
    pandas_midx = pandas.MultiIndex.from_product([['Bank_1', 'Bank_2'], ['AUD', 'CAD', 'EUR']], names=['Bank', 'Curency'])
    modin_df1, modin_df2 = (pd.DataFrame(data=data1, index=index, columns=index), pd.DataFrame(data2, modin_midx))
    pandas_df1, pandas_df2 = (pandas.DataFrame(data=data1, index=index, columns=index), pandas.DataFrame(data2, pandas_midx))
    modin_df2.columns, pandas_df2.columns = (['Notional'], ['Notional'])
    md_midx = pd.MultiIndex.from_product([modin_df2.index.levels[0], modin_df1.index])
    pd_midx = pandas.MultiIndex.from_product([pandas_df2.index.levels[0], pandas_df1.index])
    modin_result = modin_df1.reindex(md_midx, fill_value=0)
    pandas_result = pandas_df1.reindex(pd_midx, fill_value=0)
    df_equals(modin_result, pandas_result)
    modin_result = modin_df1.reindex(md_midx, fill_value=0, axis=0)
    pandas_result = pandas_df1.reindex(pd_midx, fill_value=0, axis=0)
    df_equals(modin_result, pandas_result)
    modin_result = modin_df1.reindex(md_midx, fill_value=0, axis=0, level=0)
    pandas_result = pandas_df1.reindex(pd_midx, fill_value=0, axis=0, level=0)
    df_equals(modin_result, pandas_result)