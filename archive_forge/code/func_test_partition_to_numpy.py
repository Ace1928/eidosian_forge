import io
import warnings
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test_partition_to_numpy(data):
    frame = pd.DataFrame(data)
    for partition in frame._query_compiler._modin_frame._partitions.flatten().tolist():
        assert_array_equal(partition.to_pandas().values, partition.to_numpy())