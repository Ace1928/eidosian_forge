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
def test_parquet_pos_args_deprecation(engine):
    df = pd.DataFrame({'a': [1, 2, 3]})
    msg = "Starting with pandas version 3.0 all arguments of to_parquet except for the argument 'path' will be keyword-only."
    with tm.ensure_clean() as path:
        with tm.assert_produces_warning(FutureWarning, match=msg, check_stacklevel=False, raise_on_extra_warnings=False):
            df.to_parquet(path, engine)