from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
def test_unstack_nan_index4(self):
    vals = [['Hg', np.nan, np.nan, 680585148], ['U', 0.0, np.nan, 680585148], ['Pb', 7.07e-06, np.nan, 680585148], ['Sn', 2.3614e-05, 0.0133, 680607017], ['Ag', 0.0, 0.0133, 680607017], ['Hg', -0.00015, 0.0133, 680607017]]
    df = DataFrame(vals, columns=['agent', 'change', 'dosage', 's_id'], index=[17263, 17264, 17265, 17266, 17267, 17268])
    left = df.copy().set_index(['s_id', 'dosage', 'agent']).unstack()
    vals = [[np.nan, np.nan, 7.07e-06, np.nan, 0.0], [0.0, -0.00015, np.nan, 2.3614e-05, np.nan]]
    idx = MultiIndex(levels=[[680585148, 680607017], [0.0133]], codes=[[0, 1], [-1, 0]], names=['s_id', 'dosage'])
    cols = MultiIndex(levels=[['change'], ['Ag', 'Hg', 'Pb', 'Sn', 'U']], codes=[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]], names=[None, 'agent'])
    right = DataFrame(vals, columns=cols, index=idx)
    tm.assert_frame_equal(left, right)
    left = df.loc[17264:].copy().set_index(['s_id', 'dosage', 'agent'])
    tm.assert_frame_equal(left.unstack(), right)