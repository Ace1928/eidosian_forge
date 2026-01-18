import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions
from modin.core.execution.ray.common import RayWrapper
from modin.distributed.dataframe.pandas.partitions import from_partitions
from modin.experimental.batch.pipeline import PandasQueryPipeline
from modin.tests.pandas.utils import df_equals
def test_pipeline_complex(self):
    """Create a complex pipeline with both `fan_out`, `repartition_after` and postprocessing and ensure that it runs end to end correctly."""
    from os import remove
    from os.path import exists
    from time import sleep
    df = pd.DataFrame([[0, 1, 2]])

    def new_col_adder(df, partition_id):
        sleep(60)
        df['new_col'] = partition_id
        return df

    def reducer(dfs):
        new_cols = ''.join([str(df['new_col'].values[0]) for df in dfs])
        dfs[0]['new_col1'] = new_cols
        return dfs[0]
    desired_num_partitions = 24
    pipeline = PandasQueryPipeline(df, num_partitions=desired_num_partitions)
    pipeline.add_query(new_col_adder, fan_out=True, reduce_fn=reducer, pass_partition_id=True, is_output=True, output_id=20)
    pipeline.add_query(lambda df: pandas.concat([df] * 1000), repartition_after=True)

    def to_csv(df, partition_id):
        df = df.drop(columns=['new_col'])
        df.to_csv(f'{partition_id}.csv')
        return df
    pipeline.add_query(to_csv, is_output=True, output_id=21, pass_partition_id=True)

    def post_proc(df, o_id, partition_id):
        df['new_col_proc'] = f'{o_id} {partition_id}'
        return df
    new_dfs = pipeline.compute_batch(postprocessor=post_proc, pass_partition_id=True, pass_output_id=True)
    correct_df = pd.DataFrame([[0, 1, 2]])
    correct_df['new_col'] = 0
    correct_df['new_col1'] = ''.join([str(i) for i in range(desired_num_partitions)])
    correct_df['new_col_proc'] = '20 0'
    df_equals(correct_df, new_dfs[20])
    correct_df = pd.concat([correct_df] * 1000)
    correct_df = correct_df.drop(columns=['new_col'])
    correct_df['new_col_proc'] = '21 0'
    new_length = len(correct_df.index) // desired_num_partitions
    for i in range(desired_num_partitions):
        if i == desired_num_partitions - 1:
            correct_df.iloc[i * new_length:, -1] = f'21 {i}'
        else:
            correct_df.iloc[i * new_length:(i + 1) * new_length, -1] = f'21 {i}'
    df_equals(correct_df, new_dfs[21])
    correct_df = correct_df.drop(columns=['new_col_proc'])
    for i in range(desired_num_partitions):
        if i == desired_num_partitions - 1:
            correct_partition = correct_df.iloc[i * new_length:]
        else:
            correct_partition = correct_df.iloc[i * new_length:(i + 1) * new_length]
        assert exists(f'{i}.csv'), 'CSV File for Partition {i} does not exist, even though dataframe should have been repartitioned.'
        df_equals(correct_partition, pd.read_csv(f'{i}.csv', index_col='Unnamed: 0').rename(columns={'0': 0, '1': 1, '2': 2}))
        remove(f'{i}.csv')