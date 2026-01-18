from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_rowspan_w3():
    df = DataFrame(data=[[1, 2]], index=[['l0', 'l0'], ['l1a', 'l1b']])
    styler = Styler(df, uuid='_', cell_ids=False)
    assert '<th class="row_heading level0 row0" rowspan="2">l0</th>' in styler.to_html()