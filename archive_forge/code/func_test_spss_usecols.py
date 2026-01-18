import datetime
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
def test_spss_usecols(datapath):
    fname = datapath('io', 'data', 'spss', 'labelled-num.sav')
    with pytest.raises(TypeError, match='usecols must be list-like.'):
        pd.read_spss(fname, usecols='VAR00002')