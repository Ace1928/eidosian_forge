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
def test_timestamp_nanoseconds(self, pa):
    ver = '2.6'
    df = pd.DataFrame({'a': pd.date_range('2017-01-01', freq='1ns', periods=10)})
    check_round_trip(df, pa, write_kwargs={'version': ver})