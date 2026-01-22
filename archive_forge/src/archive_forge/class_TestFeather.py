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
class TestFeather:

    def test_read_feather(self, make_feather_file):
        eval_io(fn_name='read_feather', path=make_feather_file())

    @pytest.mark.parametrize('dtype_backend', [lib.no_default, 'numpy_nullable', 'pyarrow'])
    def test_read_feather_dtype_backend(self, make_feather_file, dtype_backend):

        def comparator(df1, df2):
            df_equals(df1, df2)
            df_equals(df1.dtypes, df2.dtypes)
        eval_io(fn_name='read_feather', path=make_feather_file(), dtype_backend=dtype_backend, comparator=comparator)

    @pytest.mark.parametrize('storage_options_extra', [{'anon': False}, {'anon': True}, {'key': '123', 'secret': '123'}])
    def test_read_feather_s3(self, s3_resource, s3_storage_options, storage_options_extra):
        expected_exception = None
        if 'anon' in storage_options_extra:
            expected_exception = PermissionError('Forbidden')
        eval_io(fn_name='read_feather', path='s3://modin-test/modin-bugs/test_data.feather', storage_options=s3_storage_options | storage_options_extra, expected_exception=expected_exception)

    def test_read_feather_path_object(self, make_feather_file):
        eval_io(fn_name='read_feather', path=Path(make_feather_file()))

    def test_to_feather(self, tmp_path):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)
        eval_to_file(tmp_path, modin_obj=modin_df, pandas_obj=pandas_df, fn='to_feather', extension='feather')

    def test_read_feather_with_index_metadata(self, tmp_path):
        df = pandas.DataFrame({'a': [1, 2, 3]}, index=[0, 1, 2])
        assert not isinstance(df.index, pandas.RangeIndex)
        path = get_unique_filename(extension='.feather', data_dir=tmp_path)
        df.to_feather(path)
        eval_io(fn_name='read_feather', path=path)