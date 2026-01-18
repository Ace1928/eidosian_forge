from textwrap import dedent
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.clipboard import (
def test_round_trip_frame_string(self, df):
    df.to_clipboard(excel=False, sep=None)
    result = read_clipboard()
    assert df.to_string() == result.to_string()
    assert df.shape == result.shape