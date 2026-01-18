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
def test_unsupported_float16(self, pa):
    data = np.arange(2, 10, dtype=np.float16)
    df = pd.DataFrame(data=data, columns=['fp16'])
    if pa_version_under15p0:
        self.check_external_error_on_write(df, pa, pyarrow.ArrowException)
    else:
        check_round_trip(df, pa)