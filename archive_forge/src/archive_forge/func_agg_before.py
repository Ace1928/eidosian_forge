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
def agg_before(func, fix=False):
    """
        Run an aggregate func on the subset of data.
        """

    def _func(data):
        d = data.loc[data.index.map(lambda x: x.hour < 11)].dropna()
        if fix:
            data[data.index[0]]
        if len(d) == 0:
            return None
        return func(d)
    return _func