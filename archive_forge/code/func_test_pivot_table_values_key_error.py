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
def test_pivot_table_values_key_error():
    df = DataFrame({'eventDate': date_range(datetime.today(), periods=20, freq='ME').tolist(), 'thename': range(20)})
    df['year'] = df.set_index('eventDate').index.year
    df['month'] = df.set_index('eventDate').index.month
    with pytest.raises(KeyError, match="'badname'"):
        df.reset_index().pivot_table(index='year', columns='month', values='badname', aggfunc='count')