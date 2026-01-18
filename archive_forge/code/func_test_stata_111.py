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
def test_stata_111(self, datapath):
    df = read_stata(datapath('io', 'data', 'stata', 'stata7_111.dta'))
    original = DataFrame({'y': [1, 1, 1, 1, 1, 0, 0, np.nan, 0, 0], 'x': [1, 2, 1, 3, np.nan, 4, 3, 5, 1, 6], 'w': [2, np.nan, 5, 2, 4, 4, 3, 1, 2, 3], 'z': ['a', 'b', 'c', 'd', 'e', '', 'g', 'h', 'i', 'j']})
    original = original[['y', 'x', 'w', 'z']]
    tm.assert_frame_equal(original, df)