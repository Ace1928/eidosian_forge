import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, RangePartitioning, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
def test_sort_overpartitioned_df():
    data = [[4, 5, 6], [1, 2, 3]]
    modin_df = pd.concat([pd.DataFrame(row).T for row in data]).reset_index(drop=True)
    pandas_df = pandas.DataFrame(data)
    eval_general(modin_df, pandas_df, lambda df: df.sort_values(by=0))
    data = [list(range(100)), list(range(100, 200))]
    modin_df = pd.concat([pd.DataFrame(row).T for row in data]).reset_index(drop=True)
    pandas_df = pandas.DataFrame(data)
    eval_general(modin_df, pandas_df, lambda df: df.sort_values(by=0))
    data = np.random.choice(650, 650, replace=False).reshape((65, 10))
    modin_df = pd.concat([pd.DataFrame(row).T for row in data]).reset_index(drop=True)
    pandas_df = pandas.DataFrame(data)
    eval_general(modin_df, pandas_df, lambda df: df.sort_values(by=0))
    old_nptns = NPartitions.get()
    NPartitions.put(24)
    try:
        data = np.random.choice(650, 650, replace=False).reshape((65, 10))
        modin_df = pd.concat([pd.DataFrame(row).T for row in data]).reset_index(drop=True)
        pandas_df = pandas.DataFrame(data)
        eval_general(modin_df, pandas_df, lambda df: df.sort_values(by=0))
        data = np.random.choice(6500, 6500, replace=False).reshape((65, 100))
        modin_df = pd.concat([pd.DataFrame(row).T for row in data]).reset_index(drop=True)
        pandas_df = pandas.DataFrame(data)
        eval_general(modin_df, pandas_df, lambda df: df.sort_values(by=0))
        NPartitions.put(21)
        data = np.random.choice(6500, 6500, replace=False).reshape((65, 100))
        modin_df = pd.concat([pd.DataFrame(row).T for row in data]).reset_index(drop=True)
        pandas_df = pandas.DataFrame(data)
        eval_general(modin_df, pandas_df, lambda df: df.sort_values(by=0))
    finally:
        NPartitions.put(old_nptns)