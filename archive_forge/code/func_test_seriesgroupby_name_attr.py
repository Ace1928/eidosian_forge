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
def test_seriesgroupby_name_attr(df):
    result = df.groupby('A')['C']
    assert result.count().name == 'C'
    assert result.mean().name == 'C'
    testFunc = lambda x: np.sum(x) * 2
    assert result.agg(testFunc).name == 'C'