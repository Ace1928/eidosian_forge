import contextlib
import json
from pathlib import Path
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.experimental.pandas as pd
from modin.config import AsyncReadMode, Engine
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import try_cast_to_pandas
def test_read_csv_without_glob(self):
    with pytest.raises(FileNotFoundError):
        with warns_that_defaulting_to_pandas():
            pd.read_csv_glob('s3://dask-data/nyc-taxi/2015/yellow_tripdata_2015-', storage_options={'anon': True})