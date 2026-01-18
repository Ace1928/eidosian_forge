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
@pytest.mark.parametrize('test_async_reset_index', [False, True])
@pytest.mark.parametrize('data', [pytest.param(test_data['int_data'], marks=pytest.mark.exclude_by_default), test_data['float_nan_data']], ids=['int_data', 'float_nan_data'])
@pytest.mark.parametrize('nlevels', [3])
@pytest.mark.parametrize('level', ['no_level', None, 0, 1, 2, [2, 0], [2, 1], [1, 0], pytest.param([2, 1, 2], marks=pytest.mark.exclude_by_default), pytest.param([0, 0, 0, 0], marks=pytest.mark.exclude_by_default), pytest.param(['level_name_1'], marks=pytest.mark.exclude_by_default), pytest.param(['level_name_2', 'level_name_1'], marks=pytest.mark.exclude_by_default), pytest.param([2, 'level_name_0'], marks=pytest.mark.exclude_by_default)])
@pytest.mark.parametrize('multiindex_levels_names_max_levels', [0, 1, 2, pytest.param(3, marks=pytest.mark.exclude_by_default), pytest.param(4, marks=pytest.mark.exclude_by_default)])
@pytest.mark.parametrize('none_in_index_names', [pytest.param(False, marks=pytest.mark.exclude_by_default), True, 'mixed_1st_None', pytest.param('mixed_2nd_None', marks=pytest.mark.exclude_by_default)])
def test_reset_index_with_multi_index_drop(data, nlevels, level, multiindex_levels_names_max_levels, none_in_index_names, test_async_reset_index):
    test_reset_index_with_multi_index_no_drop(data, nlevels, True, level, 'no_col_level', 'no_col_fill', True, multiindex_levels_names_max_levels, none_in_index_names, test_async_reset_index)