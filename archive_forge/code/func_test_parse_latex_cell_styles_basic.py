from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_parse_latex_cell_styles_basic():
    cell_style = [('itshape', '--rwrap'), ('cellcolor', '[rgb]{0,1,1}--rwrap')]
    expected = '\\itshape{\\cellcolor[rgb]{0,1,1}{text}}'
    assert _parse_latex_cell_styles(cell_style, 'text') == expected