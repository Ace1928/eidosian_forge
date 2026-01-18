from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_dict_category_type(self):
    """
        Test to ensure category dtypes are maintained
        after replace with dict values
        """
    input_dict = {'col1': ['a'], 'col2': ['obj1'], 'col3': ['cat1']}
    input_df = DataFrame(data=input_dict).astype({'col1': 'category', 'col2': 'category', 'col3': 'category'})
    expected_dict = {'col1': ['z'], 'col2': ['obj9'], 'col3': ['catX']}
    expected = DataFrame(data=expected_dict).astype({'col1': 'category', 'col2': 'category', 'col3': 'category'})
    msg = 'The behavior of Series\\.replace \\(and DataFrame.replace\\) with CategoricalDtype'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = input_df.replace({'a': 'z', 'obj1': 'obj9', 'cat1': 'catX'})
    tm.assert_frame_equal(result, expected)