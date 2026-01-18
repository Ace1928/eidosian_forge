import os
import sys
import matplotlib
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.pandas.indexing import is_range_like
from modin.pandas.testing import assert_index_equal
from modin.tests.pandas.utils import (
from modin.utils import get_current_execution
def test_rename_axis_inplace():
    test_frame = pandas.DataFrame(test_data['int_data'])
    modin_df = pd.DataFrame(test_frame)
    result = test_frame.copy()
    modin_result = modin_df.copy()
    no_return = result.rename_axis('foo', inplace=True)
    modin_no_return = modin_result.rename_axis('foo', inplace=True)
    assert no_return is modin_no_return
    df_equals(modin_result, result)
    result = test_frame.copy()
    modin_result = modin_df.copy()
    no_return = result.rename_axis('bar', axis=1, inplace=True)
    modin_no_return = modin_result.rename_axis('bar', axis=1, inplace=True)
    assert no_return is modin_no_return
    df_equals(modin_result, result)