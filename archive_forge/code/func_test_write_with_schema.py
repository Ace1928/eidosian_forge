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
def test_write_with_schema(self, pa):
    import pyarrow
    df = pd.DataFrame({'x': [0, 1]})
    schema = pyarrow.schema([pyarrow.field('x', type=pyarrow.bool_())])
    out_df = df.astype(bool)
    check_round_trip(df, pa, write_kwargs={'schema': schema}, expected=out_df)