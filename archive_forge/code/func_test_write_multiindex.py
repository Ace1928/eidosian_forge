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
def test_write_multiindex(self, pa):
    engine = pa
    df = pd.DataFrame({'A': [1, 2, 3]})
    index = pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1)])
    df.index = index
    check_round_trip(df, engine)