from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def tests_merge_categorical_unordered_equal(self):
    df1 = DataFrame({'Foo': Categorical(['A', 'B', 'C'], categories=['A', 'B', 'C']), 'Left': ['A0', 'B0', 'C0']})
    df2 = DataFrame({'Foo': Categorical(['C', 'B', 'A'], categories=['C', 'B', 'A']), 'Right': ['C1', 'B1', 'A1']})
    result = merge(df1, df2, on=['Foo'])
    expected = DataFrame({'Foo': Categorical(['A', 'B', 'C']), 'Left': ['A0', 'B0', 'C0'], 'Right': ['A1', 'B1', 'C1']})
    tm.assert_frame_equal(result, expected)