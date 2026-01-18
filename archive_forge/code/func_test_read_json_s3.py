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
@pytest.mark.parametrize('storage_options_extra', [{'anon': False}, {'anon': True}, {'key': '123', 'secret': '123'}])
def test_read_json_s3(self, s3_resource, s3_storage_options, storage_options_extra):
    s3_path = 's3://modin-test/modin-bugs/test_data.json'
    expected_exception = None
    if 'anon' in storage_options_extra:
        expected_exception = PermissionError('Forbidden')
    eval_io(fn_name='read_json', path_or_buf=s3_path, lines=True, orient='records', storage_options=s3_storage_options | storage_options_extra, expected_exception=expected_exception)