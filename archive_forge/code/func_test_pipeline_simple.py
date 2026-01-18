import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions
from modin.core.execution.ray.common import RayWrapper
from modin.distributed.dataframe.pandas.partitions import from_partitions
from modin.experimental.batch.pipeline import PandasQueryPipeline
from modin.tests.pandas.utils import df_equals
def test_pipeline_simple(self):
    """Create a simple pipeline and ensure that it runs end to end correctly."""
    arr = np.random.randint(0, 1000, (1000, 1000))
    df = pd.DataFrame(arr)

    def add_col(df):
        df['new_col'] = df.sum(axis=1)
        return df
    pipeline = PandasQueryPipeline(df)
    pipeline.add_query(add_col)
    pipeline.add_query(lambda df: df * -30)
    pipeline.add_query(lambda df: df.rename(columns={i: f'col {i}' for i in range(1000)}))

    def add_row_to_partition(df):
        return pandas.concat([df, df.iloc[[-1]]])
    pipeline.add_query(add_row_to_partition, is_output=True)
    new_df = pipeline.compute_batch()[0]
    correct_df = add_col(pd.DataFrame(arr))
    correct_df *= -30
    correct_df = pd.DataFrame(correct_df.rename(columns={i: f'col {i}' for i in range(1000)})._to_pandas())
    correct_modin_frame = correct_df._query_compiler._modin_frame
    partitions = correct_modin_frame._partition_mgr_cls.row_partitions(correct_modin_frame._partitions)
    partitions = [partition.add_to_apply_calls(add_row_to_partition) for partition in partitions]
    [partition.drain_call_queue() for partition in partitions]
    partitions = [partition.list_of_blocks for partition in partitions]
    correct_df = from_partitions(partitions, axis=None)
    df_equals(correct_df, new_df)
    num_partitions = NPartitions.get()
    PandasQueryPipeline(df, num_partitions=num_partitions - 1)
    assert NPartitions.get() == num_partitions, 'Pipeline did not change NPartitions.get()'