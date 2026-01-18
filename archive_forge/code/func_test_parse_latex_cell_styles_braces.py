from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
@pytest.mark.parametrize('wrap_arg, expected', [('', '\\<command><options> <display_value>'), ('--wrap', '{\\<command><options> <display_value>}'), ('--nowrap', '\\<command><options> <display_value>'), ('--lwrap', '{\\<command><options>} <display_value>'), ('--dwrap', '{\\<command><options>}{<display_value>}'), ('--rwrap', '\\<command><options>{<display_value>}')])
def test_parse_latex_cell_styles_braces(wrap_arg, expected):
    cell_style = [('<command>', f'<options>{wrap_arg}')]
    assert _parse_latex_cell_styles(cell_style, '<display_value>') == expected