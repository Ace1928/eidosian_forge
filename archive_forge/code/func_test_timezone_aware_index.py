import datetime
from decimal import Decimal
from io import BytesIO
import os
import pathlib
import numpy as np
import pytest
from pandas._config import using_copy_on_write
from pandas._config.config import _get_option
from pandas.compat import is_platform_windows
from pandas.compat.pyarrow import (
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
from pandas.io.parquet import (
@pytest.mark.skipif(using_copy_on_write(), reason='fastparquet writes into Index')
def test_timezone_aware_index(self, fp, timezone_aware_date_list):
    idx = 5 * [timezone_aware_date_list]
    df = pd.DataFrame(index=idx, data={'index_as_col': idx})
    expected = df.copy()
    expected.index.name = 'index'
    check_round_trip(df, fp, expected=expected)