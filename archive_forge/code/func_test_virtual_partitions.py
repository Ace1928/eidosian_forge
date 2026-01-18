import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.skipif(StorageFormat.get() != 'Pandas', reason="Modin on this engine doesn't create virtual partitions.")
@pytest.mark.parametrize('left_virtual,right_virtual', [(True, False), (False, True), (True, True)])
def test_virtual_partitions(left_virtual: bool, right_virtual: bool):
    n: int = 1000
    pd_df = pandas.DataFrame(list(range(n)))

    def modin_df(is_virtual):
        if not is_virtual:
            return pd.DataFrame(pd_df)
        result = pd.concat([pd.DataFrame([i]) for i in range(n)], ignore_index=True)
        assert isinstance(result._query_compiler._modin_frame._partitions[0][0], PandasDataframeAxisPartition)
        return result
    df_equals(modin_df(left_virtual) + modin_df(right_virtual), pd_df + pd_df)