from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_short_caption(styler):
    result = styler.to_latex(caption=('full cap', 'short cap'))
    assert '\\caption[short cap]{full cap}' in result