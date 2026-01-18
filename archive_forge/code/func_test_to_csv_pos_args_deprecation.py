import io
import os
import sys
from zipfile import ZipFile
from _csv import Error
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_csv_pos_args_deprecation():
    df = DataFrame({'a': [1, 2, 3]})
    msg = "Starting with pandas version 3.0 all arguments of to_csv except for the argument 'path_or_buf' will be keyword-only."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        buffer = io.BytesIO()
        df.to_csv(buffer, ';')