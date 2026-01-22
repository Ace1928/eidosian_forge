from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def _check_setitem_invalid(self, df, invalid, indexer, warn):
    msg = 'Setting an item of incompatible dtype is deprecated'
    msg = re.escape(msg)
    orig_df = df.copy()
    with tm.assert_produces_warning(warn, match=msg):
        df.iloc[indexer, 0] = invalid
        df = orig_df.copy()
    with tm.assert_produces_warning(warn, match=msg):
        df.loc[indexer, 'a'] = invalid
        df = orig_df.copy()