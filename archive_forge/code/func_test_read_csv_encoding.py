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
@pytest.mark.parametrize('encoding', [None, 'ISO-8859-1', 'latin1', 'iso-8859-1', 'cp1252', 'utf8', pytest.param('unicode_escape', marks=pytest.mark.skipif(condition=sys.version_info < (3, 9), reason='https://bugs.python.org/issue45461')), 'raw_unicode_escape', 'utf_16_le', 'utf_16_be', 'utf32', 'utf_32_le', 'utf_32_be', 'utf-8-sig'])
def test_read_csv_encoding(self, make_csv_file, encoding):
    unique_filename = make_csv_file(encoding=encoding)
    eval_io(fn_name='read_csv', filepath_or_buffer=unique_filename, encoding=encoding)