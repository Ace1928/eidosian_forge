import bz2
import datetime as dt
from datetime import datetime
import gzip
import io
import os
import struct
import tarfile
import zipfile
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import CategoricalDtype
import pandas._testing as tm
from pandas.core.frame import (
from pandas.io.parsers import read_csv
from pandas.io.stata import (
@pytest.mark.parametrize('version', [114, 117, 118, 119, None])
def test_read_write_dta12(self, version):
    original = DataFrame([(1, 2, 3, 4, 5, 6)], columns=['astringwithmorethan32characters_1', 'astringwithmorethan32characters_2', '+', '-', 'short', 'delete'])
    formatted = DataFrame([(1, 2, 3, 4, 5, 6)], columns=['astringwithmorethan32characters_', '_0astringwithmorethan32character', '_', '_1_', '_short', '_delete'])
    formatted.index.name = 'index'
    formatted = formatted.astype(np.int32)
    with tm.ensure_clean() as path:
        with tm.assert_produces_warning(InvalidColumnName):
            original.to_stata(path, convert_dates=None, version=version)
        written_and_read_again = self.read_dta(path)
    expected = formatted.copy()
    expected.index = expected.index.astype(np.int32)
    tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)