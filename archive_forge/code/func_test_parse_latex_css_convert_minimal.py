from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
@pytest.mark.parametrize('env, inner_env', [(None, 'tabular'), ('table', 'tabular'), ('longtable', 'longtable')])
@pytest.mark.parametrize('convert, exp', [(True, 'bfseries'), (False, 'font-weightbold')])
def test_parse_latex_css_convert_minimal(styler, env, inner_env, convert, exp):
    styler.highlight_max(props='font-weight:bold;')
    result = styler.to_latex(convert_css=convert, environment=env)
    expected = dedent(f'        0 & 0 & \\{exp} -0.61 & ab \\\\\n        1 & \\{exp} 1 & -1.22 & \\{exp} cd \\\\\n        \\end{{{inner_env}}}\n    ')
    assert expected in result