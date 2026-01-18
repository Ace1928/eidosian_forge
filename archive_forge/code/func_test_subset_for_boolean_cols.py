import contextlib
import copy
import re
from textwrap import dedent
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.formats.style import (  # isort:skip
from pandas.io.formats.style_render import (
@pytest.mark.parametrize('stylefunc', ['background_gradient', 'bar', 'text_gradient'])
def test_subset_for_boolean_cols(self, stylefunc):
    df = DataFrame([[1, 2], [3, 4]], columns=[False, True])
    styled = getattr(df.style, stylefunc)()
    styled._compute()
    assert set(styled.ctx) == {(0, 0), (0, 1), (1, 0), (1, 1)}