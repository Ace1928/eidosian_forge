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
@pytest.mark.parametrize('file', ['stata4_113', 'stata4_114', 'stata4_115', 'stata4_117'])
def test_read_dta4(self, file, datapath):
    file = datapath('io', 'data', 'stata', f'{file}.dta')
    parsed = self.read_dta(file)
    expected = DataFrame.from_records([['one', 'ten', 'one', 'one', 'one'], ['two', 'nine', 'two', 'two', 'two'], ['three', 'eight', 'three', 'three', 'three'], ['four', 'seven', 4, 'four', 'four'], ['five', 'six', 5, np.nan, 'five'], ['six', 'five', 6, np.nan, 'six'], ['seven', 'four', 7, np.nan, 'seven'], ['eight', 'three', 8, np.nan, 'eight'], ['nine', 'two', 9, np.nan, 'nine'], ['ten', 'one', 'ten', np.nan, 'ten']], columns=['fully_labeled', 'fully_labeled2', 'incompletely_labeled', 'labeled_with_missings', 'float_labelled'])
    for col in expected:
        orig = expected[col].copy()
        categories = np.asarray(expected['fully_labeled'][orig.notna()])
        if col == 'incompletely_labeled':
            categories = orig
        cat = orig.astype('category')._values
        cat = cat.set_categories(categories, ordered=True)
        cat.categories.rename(None, inplace=True)
        expected[col] = cat
    tm.assert_frame_equal(parsed, expected)