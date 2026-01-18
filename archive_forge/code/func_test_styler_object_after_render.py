from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_styler_object_after_render(styler):
    pre_render = styler._copy(deepcopy=True)
    styler.to_latex(column_format='rllr', position='h', position_float='centering', hrules=True, label='my lab', caption='my cap')
    assert pre_render.table_styles == styler.table_styles
    assert pre_render.caption == styler.caption