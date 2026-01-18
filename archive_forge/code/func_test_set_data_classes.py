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
@pytest.mark.parametrize('classes', [DataFrame(data=[['', 'test-class'], [np.nan, None]], columns=['A', 'B'], index=['a', 'b']), DataFrame(data=[['test-class']], columns=['B'], index=['a']), DataFrame(data=[['test-class', 'unused']], columns=['B', 'C'], index=['a'])])
def test_set_data_classes(self, classes):
    df = DataFrame(data=[[0, 1], [2, 3]], columns=['A', 'B'], index=['a', 'b'])
    s = Styler(df, uuid_len=0, cell_ids=False).set_td_classes(classes).to_html()
    assert '<td class="data row0 col0" >0</td>' in s
    assert '<td class="data row0 col1 test-class" >1</td>' in s
    assert '<td class="data row1 col0" >2</td>' in s
    assert '<td class="data row1 col1" >3</td>' in s
    s = Styler(df, uuid_len=0, cell_ids=True).set_td_classes(classes).to_html()
    assert '<td id="T__row0_col0" class="data row0 col0" >0</td>' in s
    assert '<td id="T__row0_col1" class="data row0 col1 test-class" >1</td>' in s
    assert '<td id="T__row1_col0" class="data row1 col0" >2</td>' in s
    assert '<td id="T__row1_col1" class="data row1 col1" >3</td>' in s