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
def test_pickles(datapath):
    if not is_platform_little_endian():
        pytest.skip('known failure on non-little endian')
    for legacy_pickle in Path(__file__).parent.glob('data/legacy_pickle/*/*.p*kl*'):
        legacy_pickle = datapath(legacy_pickle)
        data = pd.read_pickle(legacy_pickle)
        for typ, dv in data.items():
            for dt, result in dv.items():
                expected = data[typ][dt]
                if typ == 'series' and dt == 'ts':
                    tm.assert_series_equal(result, expected)
                    assert result.index.freq == expected.index.freq
                    assert not result.index.freq.normalize
                    tm.assert_series_equal(result > 0, expected > 0)
                    freq = result.index.freq
                    assert freq + Day(1) == Day(2)
                    res = freq + pd.Timedelta(hours=1)
                    assert isinstance(res, pd.Timedelta)
                    assert res == pd.Timedelta(days=1, hours=1)
                    res = freq + pd.Timedelta(nanoseconds=1)
                    assert isinstance(res, pd.Timedelta)
                    assert res == pd.Timedelta(days=1, nanoseconds=1)
                elif typ == 'index' and dt == 'period':
                    tm.assert_index_equal(result, expected)
                    assert isinstance(result.freq, MonthEnd)
                    assert result.freq == MonthEnd()
                    assert result.freqstr == 'M'
                    tm.assert_index_equal(result.shift(2), expected.shift(2))
                elif typ == 'series' and dt in ('dt_tz', 'cat'):
                    tm.assert_series_equal(result, expected)
                elif typ == 'frame' and dt in ('dt_mixed_tzs', 'cat_onecol', 'cat_and_float'):
                    tm.assert_frame_equal(result, expected)
                else:
                    compare_element(result, expected, typ)