from io import (
import os
import platform
from urllib.error import URLError
import uuid
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_path_path_lib(all_parsers):
    parser = all_parsers
    df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
    result = tm.round_trip_pathlib(df.to_csv, lambda p: parser.read_csv(p, index_col=0))
    tm.assert_frame_equal(df, result)