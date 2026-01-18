from datetime import (
from functools import partial
from io import BytesIO
import os
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
from pandas.compat._constants import PY310
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._util import _writers
@pytest.mark.parametrize('dtype', [None, object])
def test_raise_when_saving_timezones(self, dtype, tz_aware_fixture, path):
    tz = tz_aware_fixture
    data = pd.Timestamp('2019', tz=tz)
    df = DataFrame([data], dtype=dtype)
    with pytest.raises(ValueError, match='Excel does not support'):
        df.to_excel(path)
    data = data.to_pydatetime()
    df = DataFrame([data], dtype=dtype)
    with pytest.raises(ValueError, match='Excel does not support'):
        df.to_excel(path)