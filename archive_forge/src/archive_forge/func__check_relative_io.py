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
def _check_relative_io(fn_name, unique_filename, path_arg, storage_default=()):
    dirname, basename = os.path.split(unique_filename)
    pinned_home = {envvar: dirname for envvar in ('HOME', 'USERPROFILE', 'HOMEPATH')}
    should_default = Engine.get() == 'Python' or StorageFormat.get() in storage_default
    with mock.patch.dict(os.environ, pinned_home):
        with warns_that_defaulting_to_pandas() if should_default else contextlib.nullcontext():
            eval_io(fn_name=fn_name, **{path_arg: f'~/{basename}'})
        eval_general(f'~/{basename}', unique_filename, lambda fname: getattr(pandas, fn_name)(**{path_arg: fname}))