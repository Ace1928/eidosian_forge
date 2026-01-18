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
def test_read_gbq_mock(self):
    test_args = ('fake_query',)
    test_kwargs = inspect.signature(pd.read_gbq).parameters.copy()
    test_kwargs.update(project_id='test_id', dialect='standart')
    test_kwargs.pop('query', None)
    with mock.patch('pandas.read_gbq', return_value=pandas.DataFrame([])) as read_gbq:
        pd.read_gbq(*test_args, **test_kwargs)
    read_gbq.assert_called_once_with(*test_args, **test_kwargs)