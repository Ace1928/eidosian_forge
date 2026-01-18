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
@pytest.mark.parametrize('file', ['stata1_114', 'stata1_117'])
def test_read_dta1(self, file, datapath):
    file = datapath('io', 'data', 'stata', f'{file}.dta')
    parsed = self.read_dta(file)
    expected = DataFrame([(np.nan, np.nan, np.nan, np.nan, np.nan)], columns=['float_miss', 'double_miss', 'byte_miss', 'int_miss', 'long_miss'])
    expected['float_miss'] = expected['float_miss'].astype(np.float32)
    tm.assert_frame_equal(parsed, expected)