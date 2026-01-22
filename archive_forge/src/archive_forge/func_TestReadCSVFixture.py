import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Optional
import boto3
import numpy as np
import pandas
import pytest
import requests
import s3fs
from pandas.util._decorators import doc
import modin.utils  # noqa: E402
import uuid  # noqa: E402
import modin  # noqa: E402
import modin.config  # noqa: E402
import modin.tests.config  # noqa: E402
from modin.config import (  # noqa: E402
from modin.core.execution.dispatching.factories import factories  # noqa: E402
from modin.core.execution.python.implementations.pandas_on_python.io import (  # noqa: E402
from modin.core.storage_formats import (  # noqa: E402
from modin.tests.pandas.utils import (  # noqa: E402
@pytest.fixture(scope='class')
def TestReadCSVFixture(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('TestReadCSVFixture')
    creator = _make_csv_file(data_dir=tmp_path)
    pytest.csvs_names = {}
    pytest.csvs_names['test_read_csv_regular'] = creator()
    pytest.csvs_names['test_read_csv_yes_no'] = creator(additional_col_values=['Yes', 'true', 'No', 'false'])
    pytest.csvs_names['test_read_csv_blank_lines'] = creator(add_blank_lines=True)
    pytest.csvs_names['test_read_csv_nans'] = creator(add_blank_lines=True, additional_col_values=['<NA>', 'N/A', 'NA', 'NULL', 'custom_nan', '73'])
    pytest.csvs_names['test_read_csv_bad_lines'] = creator(add_bad_lines=True)
    yield