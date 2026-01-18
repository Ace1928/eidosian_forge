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
def test_read_pickle_with_subclass():
    expected = (Series(dtype=object), MyTz())
    result = tm.round_trip_pickle(expected)
    tm.assert_series_equal(result[0], expected[0])
    assert isinstance(result[1], MyTz)