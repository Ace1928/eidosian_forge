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
def test_read_write_dta11(self):
    original = DataFrame([(1, 2, 3, 4)], columns=['good', 'bäd', '8number', 'astringwithmorethan32characters______'])
    formatted = DataFrame([(1, 2, 3, 4)], columns=['good', 'b_d', '_8number', 'astringwithmorethan32characters_'])
    formatted.index.name = 'index'
    formatted = formatted.astype(np.int32)
    with tm.ensure_clean() as path:
        with tm.assert_produces_warning(InvalidColumnName):
            original.to_stata(path, convert_dates=None)
        written_and_read_again = self.read_dta(path)
    expected = formatted.copy()
    expected.index = expected.index.astype(np.int32)
    tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)