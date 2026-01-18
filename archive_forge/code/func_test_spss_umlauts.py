import datetime
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.filterwarnings('ignore::pandas.errors.ChainedAssignmentError')
@pytest.mark.filterwarnings('ignore:ChainedAssignmentError:FutureWarning')
def test_spss_umlauts(datapath):
    fname = datapath('io', 'data', 'spss', 'umlauts.sav')
    df = pd.read_spss(fname, convert_categoricals=True)
    expected = pd.DataFrame({'var1': ['the ä umlaut', 'the ü umlaut', 'the ä umlaut', 'the ö umlaut']})
    expected['var1'] = pd.Categorical(expected['var1'])
    tm.assert_frame_equal(df, expected)
    df = pd.read_spss(fname, convert_categoricals=False)
    expected = pd.DataFrame({'var1': [1.0, 2.0, 1.0, 3.0]})
    tm.assert_frame_equal(df, expected)