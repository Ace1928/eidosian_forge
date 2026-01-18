from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
@pytest.mark.parametrize('index', [Index(range(5)), date_range('2020-01-01', periods=10), MultiIndex.from_tuples([('foo', '1'), ('bar', '3')]), period_range(start='2000', end='2010', freq='Y')])
def test_str_attribute_raises(self, index):
    with pytest.raises(AttributeError, match='only use .str accessor'):
        index.str.repeat(2)