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
@pytest.mark.parametrize('engine', ['c'])
@pytest.mark.parametrize('delimiter', [',', ' '])
@pytest.mark.parametrize('low_memory', [True, False])
@pytest.mark.parametrize('memory_map', [True, False])
@pytest.mark.parametrize('float_precision', [None, 'high', 'round_trip'])
def test_read_csv_internal(self, make_csv_file, engine, delimiter, low_memory, memory_map, float_precision):
    unique_filename = make_csv_file(delimiter=delimiter)
    eval_io(filepath_or_buffer=unique_filename, fn_name='read_csv', engine=engine, delimiter=delimiter, low_memory=low_memory, memory_map=memory_map, float_precision=float_precision)