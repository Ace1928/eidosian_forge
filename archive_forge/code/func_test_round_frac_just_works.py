import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
@pytest.mark.parametrize('x', [np.arange(11.0), np.arange(11.0) / 10000000000.0])
def test_round_frac_just_works(x):
    cut(x, 2)