from __future__ import annotations
from array import array
import bz2
import datetime
import functools
from functools import partial
import gzip
import io
import os
from pathlib import Path
import pickle
import shutil
import tarfile
from typing import Any
import uuid
import zipfile
import numpy as np
import pytest
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
from pandas.compat.compressors import flatten_buffer
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.generate_legacy_storage_files import create_pickle_data
import pandas.io.common as icom
from pandas.tseries.offsets import (
@pytest.mark.parametrize('pickle_writer', [pytest.param(python_pickler, id='python'), pytest.param(pd.to_pickle, id='pandas_proto_default'), pytest.param(functools.partial(pd.to_pickle, protocol=pickle.HIGHEST_PROTOCOL), id='pandas_proto_highest'), pytest.param(functools.partial(pd.to_pickle, protocol=4), id='pandas_proto_4'), pytest.param(functools.partial(pd.to_pickle, protocol=5), id='pandas_proto_5')])
@pytest.mark.parametrize('writer', [pd.to_pickle, python_pickler])
@pytest.mark.parametrize('typ, expected', flatten(create_pickle_data()))
def test_round_trip_current(typ, expected, pickle_writer, writer):
    with tm.ensure_clean() as path:
        pickle_writer(expected, path)
        result = pd.read_pickle(path)
        compare_element(result, expected, typ)
        result = python_unpickler(path)
        compare_element(result, expected, typ)
        with open(path, mode='wb') as handle:
            writer(expected, path)
            handle.seek(0)
        with open(path, mode='rb') as handle:
            result = pd.read_pickle(handle)
            handle.seek(0)
        compare_element(result, expected, typ)