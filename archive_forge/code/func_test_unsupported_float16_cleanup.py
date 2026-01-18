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
@pytest.mark.xfail(is_platform_windows(), reason='PyArrow does not cleanup of partial files dumps when unsupported dtypes are passed to_parquet function in windows')
@pytest.mark.skipif(not pa_version_under15p0, reason='float16 works on 15')
@pytest.mark.parametrize('path_type', [str, pathlib.Path])
def test_unsupported_float16_cleanup(self, pa, path_type):
    data = np.arange(2, 10, dtype=np.float16)
    df = pd.DataFrame(data=data, columns=['fp16'])
    with tm.ensure_clean() as path_str:
        path = path_type(path_str)
        with tm.external_error_raised(pyarrow.ArrowException):
            df.to_parquet(path=path, engine=pa)
        assert not os.path.isfile(path)