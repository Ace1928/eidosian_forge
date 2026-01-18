from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_obj_arg_get_group_deprecated():
    depr_msg = 'obj is deprecated'
    df = DataFrame({'a': [1, 1, 2], 'b': [3, 4, 5]})
    expected = df.iloc[df.groupby('b').indices.get(4)]
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        result = df.groupby('b').get_group(4, obj=df)
        tm.assert_frame_equal(result, expected)