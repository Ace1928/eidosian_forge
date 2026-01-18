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
@pytest.mark.parametrize('storage_options', [{'anon': False}, {'anon': True}, {'key': '123', 'secret': '123'}, None])
@pytest.mark.xfail(reason='S3 file gone missing, see https://github.com/modin-project/modin/issues/4875')
def test_read_csv_s3(self, storage_options):
    eval_io(fn_name='read_csv', filepath_or_buffer='s3://noaa-ghcn-pds/csv/1788.csv', storage_options=storage_options)