from textwrap import dedent
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.clipboard import (
def test_read_clipboard_infer_excel(self, clipboard):
    clip_kwargs = {'engine': 'python'}
    text = dedent('\n            John James\tCharlie Mingus\n            1\t2\n            4\tHarry Carney\n            '.strip())
    clipboard.setText(text)
    df = read_clipboard(**clip_kwargs)
    assert df.iloc[1, 1] == 'Harry Carney'
    text = dedent('\n            a\t b\n            1  2\n            3  4\n            '.strip())
    clipboard.setText(text)
    res = read_clipboard(**clip_kwargs)
    text = dedent('\n            a  b\n            1  2\n            3  4\n            '.strip())
    clipboard.setText(text)
    exp = read_clipboard(**clip_kwargs)
    tm.assert_frame_equal(res, exp)