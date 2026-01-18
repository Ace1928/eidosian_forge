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
@pytest.mark.parametrize('sep', ['_', ',', '.'])
@pytest.mark.parametrize('decimal', ['.', '_'])
@pytest.mark.parametrize('thousands', [None, ',', '_', ' '])
def test_read_csv_seps(self, make_csv_file, sep, decimal, thousands):
    unique_filename = make_csv_file(delimiter=sep, thousands_separator=thousands, decimal_separator=decimal)
    eval_io(fn_name='read_csv', filepath_or_buffer=unique_filename, sep=sep, decimal=decimal, thousands=thousands)