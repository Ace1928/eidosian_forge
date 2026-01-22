import contextlib
import csv
import inspect
import os
import sys
import unittest.mock as mock
from collections import defaultdict
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict
import fastparquet
import numpy as np
import pandas
import pandas._libs.lib as lib
import pyarrow as pa
import pyarrow.dataset
import pytest
import sqlalchemy as sa
from packaging import version
from pandas._testing import ensure_clean
from pandas.errors import ParserWarning
from scipy import sparse
from modin.config import (
from modin.db_conn import ModinDatabaseConnection, UnsupportedDatabaseException
from modin.pandas.io import from_arrow, from_dask, from_ray, to_pandas
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from .utils import (
from .utils import test_data as utils_test_data
from .utils import time_parsing_csv_path
from modin.config import NPartitions
@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestClipboard:

    @pytest.mark.skip(reason='No clipboard in CI')
    def test_read_clipboard(self):
        setup_clipboard()
        eval_io(fn_name='read_clipboard')

    @pytest.mark.skip(reason='No clipboard in CI')
    def test_to_clipboard(self):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)
        modin_df.to_clipboard()
        modin_as_clip = pandas.read_clipboard()
        pandas_df.to_clipboard()
        pandas_as_clip = pandas.read_clipboard()
        assert modin_as_clip.equals(pandas_as_clip)