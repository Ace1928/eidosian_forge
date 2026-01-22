import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def _check_diff_index(df_list, result, exp_index):
    reindexed = [x.reindex(exp_index) for x in df_list]
    expected = reindexed[0].join(reindexed[1:])
    tm.assert_frame_equal(result, expected)