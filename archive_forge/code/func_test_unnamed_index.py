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
def test_unnamed_index(self):

    def get_internal_df(df):
        partition = read_df._query_compiler._modin_frame._partitions[0][0]
        return partition.to_pandas()
    path = 'modin/tests/pandas/data/issue_3119.csv'
    read_df = pd.read_csv(path, index_col=0)
    assert get_internal_df(read_df).index.name is None
    read_df = pd.read_csv(path, index_col=[0, 1])
    for name1, name2 in zip(get_internal_df(read_df).index.names, [None, 'a']):
        assert name1 == name2