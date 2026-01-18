import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions
from modin.core.execution.ray.common import RayWrapper
from modin.distributed.dataframe.pandas.partitions import from_partitions
from modin.experimental.batch.pipeline import PandasQueryPipeline
from modin.tests.pandas.utils import df_equals
def test_postprocessing_with_output_id(self):
    """Check that the `postprocessor` argument is correctly handled when `output_id` is specified."""

    def new_col_adder(df):
        df['new_col'] = df.iloc[:, -1]
        return df
    arr = np.random.randint(0, 1000, (1000, 1000))
    df = pd.DataFrame(arr)
    pipeline = PandasQueryPipeline(df)
    pipeline.add_query(lambda df: df * -30, is_output=True, output_id=20)
    pipeline.add_query(lambda df: df.rename(columns={i: f'col {i}' for i in range(1000)}), is_output=True, output_id=21)
    pipeline.add_query(lambda df: df + 30, is_output=True, output_id=22)
    new_dfs = pipeline.compute_batch(postprocessor=new_col_adder)
    assert len(new_dfs) == 3, 'Pipeline did not return all outputs'