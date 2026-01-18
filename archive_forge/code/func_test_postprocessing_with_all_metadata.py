import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions
from modin.core.execution.ray.common import RayWrapper
from modin.distributed.dataframe.pandas.partitions import from_partitions
from modin.experimental.batch.pipeline import PandasQueryPipeline
from modin.tests.pandas.utils import df_equals
def test_postprocessing_with_all_metadata(self):
    """Check that postprocessing is correctly handled when `partition_id` and `output_id` are passed."""
    arr = np.random.randint(0, 1000, (1000, 1000))

    def new_col_adder(df, o_id, partition_id):
        df['new_col'] = f'{o_id} {partition_id}'
        return df
    df = pd.DataFrame(arr)
    pipeline = PandasQueryPipeline(df)
    pipeline.add_query(lambda df: df * -30, is_output=True, output_id=20)
    pipeline.add_query(lambda df: df.rename(columns={i: f'col {i}' for i in range(1000)}), is_output=True, output_id=21)
    new_dfs = pipeline.compute_batch(postprocessor=new_col_adder, pass_partition_id=True, pass_output_id=True)
    correct_df = pd.DataFrame(arr) * -30
    correct_modin_frame = correct_df._query_compiler._modin_frame
    partitions = correct_modin_frame._partition_mgr_cls.row_partitions(correct_modin_frame._partitions)
    partitions = [partition.add_to_apply_calls(new_col_adder, 20, i) for i, partition in enumerate(partitions)]
    [partition.drain_call_queue() for partition in partitions]
    partitions = [partition.list_of_blocks for partition in partitions]
    correct_df = from_partitions(partitions, axis=None)
    df_equals(correct_df, new_dfs[20])
    correct_df = correct_df.drop(columns=['new_col'])
    correct_df = pd.DataFrame(correct_df.rename(columns={i: f'col {i}' for i in range(1000)})._to_pandas())
    correct_modin_frame = correct_df._query_compiler._modin_frame
    partitions = correct_modin_frame._partition_mgr_cls.row_partitions(correct_modin_frame._partitions)
    partitions = [partition.add_to_apply_calls(new_col_adder, 21, i) for i, partition in enumerate(partitions)]
    [partition.drain_call_queue() for partition in partitions]
    partitions = [partition.list_of_blocks for partition in partitions]
    correct_df = from_partitions(partitions, axis=None)
    df_equals(correct_df, new_dfs[21])