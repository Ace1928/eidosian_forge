import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
def test_quantile_7157():
    n_rows = 100
    n_fcols = 10
    n_mcols = 5
    df1_md, df1_pd = create_test_dfs(random_state.rand(n_rows, n_fcols), columns=[f'feat_{i}' for i in range(n_fcols)])
    df2_md, df2_pd = create_test_dfs({'test_string1': ['test_string2' for _ in range(n_rows)] for _ in range(n_mcols)})
    df3_md = pd.concat([df2_md, df1_md], axis=1)
    df3_pd = pandas.concat([df2_pd, df1_pd], axis=1)
    eval_general(df3_md, df3_pd, lambda df: df.quantile(0.25, numeric_only=True))
    eval_general(df3_md, df3_pd, lambda df: df.quantile((0.25,), numeric_only=True))
    eval_general(df3_md, df3_pd, lambda df: df.quantile((0.25, 0.75), numeric_only=True))