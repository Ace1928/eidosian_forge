import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions
from modin.core.execution.ray.common import RayWrapper
from modin.distributed.dataframe.pandas.partitions import from_partitions
from modin.experimental.batch.pipeline import PandasQueryPipeline
from modin.tests.pandas.utils import df_equals
def test_postprocessing(self):
    """Check that the `postprocessor` argument to `_compute_batch` is handled correctly."""
    arr = np.random.randint(0, 1000, (1000, 1000))
    df = pd.DataFrame(arr)
    pipeline = PandasQueryPipeline(df)
    pipeline.add_query(lambda df: df * -30, is_output=True)
    pipeline.add_query(lambda df: df.rename(columns={i: f'col {i}' for i in range(1000)}), is_output=True)
    pipeline.add_query(lambda df: df + 30, is_output=True)

    def new_col_adder(df):
        df['new_col'] = df.iloc[:, -1]
        return df
    new_dfs = pipeline.compute_batch(postprocessor=new_col_adder)
    assert len(new_dfs) == 3, 'Pipeline did not return all outputs'
    correct_df = pd.DataFrame(arr) * -30
    correct_df['new_col'] = correct_df.iloc[:, -1]
    df_equals(correct_df, new_dfs[0])
    correct_df = correct_df.drop(columns=['new_col'])
    correct_df = correct_df.rename(columns={i: f'col {i}' for i in range(1000)})
    correct_df['new_col'] = correct_df.iloc[:, -1]
    df_equals(correct_df, new_dfs[1])
    correct_df = correct_df.drop(columns=['new_col'])
    correct_df += 30
    correct_df['new_col'] = correct_df.iloc[:, -1]
    df_equals(correct_df, new_dfs[2])