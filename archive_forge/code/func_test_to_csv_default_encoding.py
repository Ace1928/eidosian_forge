import io
import os
import sys
from zipfile import ZipFile
from _csv import Error
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_csv_default_encoding(self):
    df = DataFrame({'col': ['AAAAA', 'ÄÄÄÄÄ', 'ßßßßß', '聞聞聞聞聞']})
    with tm.ensure_clean('test.csv') as path:
        df.to_csv(path)
        tm.assert_frame_equal(pd.read_csv(path, index_col=0), df)