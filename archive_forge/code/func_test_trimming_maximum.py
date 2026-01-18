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
@pytest.mark.parametrize('rn, cn, max_els, max_rows, max_cols, exp_rn, exp_cn', [(100, 100, 100, None, None, 12, 6), (1000, 3, 750, None, None, 250, 3), (4, 1000, 500, None, None, 4, 125), (1000, 3, 750, 10, None, 10, 3), (4, 1000, 500, None, 5, 4, 5), (100, 100, 700, 50, 50, 25, 25)])
def test_trimming_maximum(rn, cn, max_els, max_rows, max_cols, exp_rn, exp_cn):
    rn, cn = _get_trimming_maximums(rn, cn, max_els, max_rows, max_cols, scaling_factor=0.5)
    assert (rn, cn) == (exp_rn, exp_cn)