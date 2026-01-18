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
def test_rename_bug():
    frame_data = {0: ['foo', 'bar'], 1: ['bah', 'bas'], 2: [1, 2]}
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    df = df.rename(columns={0: 'a'})
    df = df.rename(columns={1: 'b'})
    df = df.set_index(['a', 'b'])
    df.columns = ['2001-01-01']
    modin_df = modin_df.rename(columns={0: 'a'})
    modin_df = modin_df.rename(columns={1: 'b'})
    modin_df = modin_df.set_index(['a', 'b'])
    modin_df.columns = ['2001-01-01']
    df_equals(modin_df, df)