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
def _check_groupby(df, result, keys, field, f=lambda x: x.sum()):
    tups = [tuple(row) for row in df[keys].values]
    tups = com.asarray_tuplesafe(tups)
    expected = f(df.groupby(tups)[field])
    for k, v in expected.items():
        assert result[k] == v