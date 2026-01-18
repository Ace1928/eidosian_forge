import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sasreader import read_sas
def test_truncated_float_support(self, file04):
    data_csv = pd.read_csv(file04.replace('.xpt', '.csv'))
    data = read_sas(file04, format='xport')
    tm.assert_frame_equal(data.astype('int64'), data_csv)