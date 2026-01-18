import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions
from modin.core.execution.ray.common import RayWrapper
from modin.distributed.dataframe.pandas.partitions import from_partitions
from modin.experimental.batch.pipeline import PandasQueryPipeline
from modin.tests.pandas.utils import df_equals
def test_postprocessing_with_output_id_passed(self):
    """Check that the `postprocessor` argument is correctly passed `output_id` when `pass_output_id` is `True`."""
    arr = np.random.randint(0, 1000, (1000, 1000))

    def new_col_adder(df, o_id):
        df['new_col'] = o_id
        return df
    df = pd.DataFrame(arr)
    pipeline = PandasQueryPipeline(df)
    pipeline.add_query(lambda df: df * -30, is_output=True, output_id=20)
    pipeline.add_query(lambda df: df.rename(columns={i: f'col {i}' for i in range(1000)}), is_output=True, output_id=21)
    pipeline.add_query(lambda df: df + 30, is_output=True, output_id=22)
    new_dfs = pipeline.compute_batch(postprocessor=new_col_adder, pass_output_id=True)
    correct_df = pd.DataFrame(arr) * -30
    correct_df['new_col'] = 20
    df_equals(correct_df, new_dfs[20])
    correct_df = correct_df.drop(columns=['new_col'])
    correct_df = correct_df.rename(columns={i: f'col {i}' for i in range(1000)})
    correct_df['new_col'] = 21
    df_equals(correct_df, new_dfs[21])
    correct_df = correct_df.drop(columns=['new_col'])
    correct_df += 30
    correct_df['new_col'] = 22
    df_equals(correct_df, new_dfs[22])