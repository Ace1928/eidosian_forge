import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_unneeded_quoting(self, df):
    res = df.query('`A` > 2')
    expect = df[df['A'] > 2]
    tm.assert_frame_equal(res, expect)