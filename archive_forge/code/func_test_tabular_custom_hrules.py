from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_tabular_custom_hrules(styler):
    styler.set_table_styles([{'selector': 'toprule', 'props': ':hline'}, {'selector': 'bottomrule', 'props': ':otherline'}])
    expected = dedent('        \\begin{tabular}{lrrl}\n        \\hline\n         & A & B & C \\\\\n        0 & 0 & -0.61 & ab \\\\\n        1 & 1 & -1.22 & cd \\\\\n        \\otherline\n        \\end{tabular}\n        ')
    assert styler.to_latex() == expected