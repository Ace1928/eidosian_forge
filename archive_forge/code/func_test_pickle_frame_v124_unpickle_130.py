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
def test_pickle_frame_v124_unpickle_130(datapath):
    path = datapath(Path(__file__).parent, 'data', 'legacy_pickle', '1.2.4', 'empty_frame_v1_2_4-GH#42345.pkl')
    with open(path, 'rb') as fd:
        df = pickle.load(fd)
    expected = DataFrame(index=[], columns=[])
    tm.assert_frame_equal(df, expected)