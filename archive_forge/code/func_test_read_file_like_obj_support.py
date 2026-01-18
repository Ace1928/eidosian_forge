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
def test_read_file_like_obj_support(self, df_compat):
    pytest.importorskip('pyarrow')
    buffer = BytesIO()
    df_compat.to_parquet(buffer)
    df_from_buf = read_parquet(buffer)
    tm.assert_frame_equal(df_compat, df_from_buf)