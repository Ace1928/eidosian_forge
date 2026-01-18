import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
def test_relabel_roundtrip(styler):
    styler.relabel_index(['{}', '{}'])
    ctx = styler._translate(True, True)
    assert {'value': 'x', 'display_value': 'x'}.items() <= ctx['body'][0][0].items()
    assert {'value': 'y', 'display_value': 'y'}.items() <= ctx['body'][1][0].items()