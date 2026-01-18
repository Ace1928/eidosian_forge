import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions
from modin.core.execution.ray.common import RayWrapper
from modin.distributed.dataframe.pandas.partitions import from_partitions
from modin.experimental.batch.pipeline import PandasQueryPipeline
from modin.tests.pandas.utils import df_equals
@pytest.mark.skipif(Engine.get() == 'Ray', reason='Ray supports the Batch Pipeline API')
def test_pipeline_unsupported_engine():
    """Ensure that trying to use the Pipeline API with an unsupported Engine raises errors."""
    df = pd.DataFrame([[1]])
    with pytest.raises(NotImplementedError, match='Batch Pipeline API is only implemented for `PandasOnRay` execution.'):
        PandasQueryPipeline(df)
    eng = Engine.get()
    Engine.put('Ray')
    with pytest.raises(NotImplementedError, match='Batch Pipeline API is only implemented for `PandasOnRay` execution.'):
        PandasQueryPipeline(df, 0)
    df_on_ray_engine = pd.DataFrame([[1]])
    pipeline = PandasQueryPipeline(df_on_ray_engine)
    with pytest.raises(NotImplementedError, match='Batch Pipeline API is only implemented for `PandasOnRay` execution.'):
        pipeline.update_df(df)
    Engine.put(eng)
    with pytest.raises(NotImplementedError, match='Batch Pipeline API is only implemented for `PandasOnRay` execution.'):
        pipeline.update_df(df)