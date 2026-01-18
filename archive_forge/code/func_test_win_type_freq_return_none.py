import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
def test_win_type_freq_return_none():
    freq_roll = Series(range(2), index=date_range('2020', periods=2)).rolling('2s')
    assert freq_roll.win_type is None