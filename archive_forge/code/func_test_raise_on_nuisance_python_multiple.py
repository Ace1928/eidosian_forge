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
def test_raise_on_nuisance_python_multiple(three_group):
    grouped = three_group.groupby(['A', 'B'])
    msg = re.escape('agg function failed [how->mean,dtype->')
    with pytest.raises(TypeError, match=msg):
        grouped.agg('mean')
    with pytest.raises(TypeError, match=msg):
        grouped.mean()