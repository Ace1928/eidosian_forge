import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
def test_relabel_columns(styler_multi):
    labels = [(1, 2), (3, 4)]
    styler_multi.hide(axis=1, subset=[('A', 'a'), ('B', 'b')])
    styler_multi.relabel_index(axis=1, labels=labels)
    ctx = styler_multi._translate(True, True)
    assert {'value': 'A', 'display_value': 1}.items() <= ctx['head'][0][3].items()
    assert {'value': 'B', 'display_value': 3}.items() <= ctx['head'][0][4].items()
    assert {'value': 'b', 'display_value': 2}.items() <= ctx['head'][1][3].items()
    assert {'value': 'a', 'display_value': 4}.items() <= ctx['head'][1][4].items()