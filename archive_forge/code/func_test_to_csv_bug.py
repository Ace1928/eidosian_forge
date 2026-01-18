import csv
from io import StringIO
import os
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle
def test_to_csv_bug(self):
    f1 = StringIO('a,1.0\nb,2.0')
    df = self.read_csv(f1, header=None)
    newdf = DataFrame({'t': df[df.columns[0]]})
    with tm.ensure_clean() as path:
        newdf.to_csv(path)
        recons = read_csv(path, index_col=0)
        tm.assert_frame_equal(recons, newdf, check_names=False)