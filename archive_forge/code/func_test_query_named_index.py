import matplotlib
import numpy as np
import pandas
import pytest
from pandas.core.dtypes.common import is_list_like
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
def test_query_named_index():
    eval_general(*(df.set_index('col1') for df in create_test_dfs(test_data['int_data'])), lambda df: df.query('col1 % 2 == 0 | col3 % 2 == 1'))