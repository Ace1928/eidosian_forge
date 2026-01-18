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
def test_write_missing_strings(self):
    original = DataFrame([['1'], [None]], columns=['foo'])
    expected = DataFrame([['1'], ['']], index=pd.Index([0, 1], dtype=np.int32, name='index'), columns=['foo'])
    with tm.ensure_clean() as path:
        original.to_stata(path)
        written_and_read_again = self.read_dta(path)
    tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)