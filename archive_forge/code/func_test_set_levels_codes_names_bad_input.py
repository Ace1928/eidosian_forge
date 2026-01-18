import numpy as np
import pytest
from pandas.compat import PY311
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_set_levels_codes_names_bad_input(idx):
    levels, codes = (idx.levels, idx.codes)
    names = idx.names
    with pytest.raises(ValueError, match='Length of levels'):
        idx.set_levels([levels[0]])
    with pytest.raises(ValueError, match='Length of codes'):
        idx.set_codes([codes[0]])
    with pytest.raises(ValueError, match='Length of names'):
        idx.set_names([names[0]])
    with pytest.raises(TypeError, match='list of lists-like'):
        idx.set_levels(levels[0])
    with pytest.raises(TypeError, match='list of lists-like'):
        idx.set_codes(codes[0])
    with pytest.raises(TypeError, match='list-like'):
        idx.set_names(names[0])
    with pytest.raises(TypeError, match='list of lists-like'):
        idx.set_levels(levels[0], level=[0, 1])
    with pytest.raises(TypeError, match='list-like'):
        idx.set_levels(levels, level=0)
    with pytest.raises(TypeError, match='list of lists-like'):
        idx.set_codes(codes[0], level=[0, 1])
    with pytest.raises(TypeError, match='list-like'):
        idx.set_codes(codes, level=0)
    with pytest.raises(ValueError, match='Length of names'):
        idx.set_names(names[0], level=[0, 1])
    with pytest.raises(TypeError, match='Names must be a'):
        idx.set_names(names, level=0)