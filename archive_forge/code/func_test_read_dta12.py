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
def test_read_dta12(self, datapath):
    parsed_117 = self.read_dta(datapath('io', 'data', 'stata', 'stata12_117.dta'))
    expected = DataFrame.from_records([[1, 'abc', 'abcdefghi'], [3, 'cba', 'qwertywertyqwerty'], [93, '', 'strl']], columns=['x', 'y', 'z'])
    tm.assert_frame_equal(parsed_117, expected, check_dtype=False)