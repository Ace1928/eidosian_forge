from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_siunitx_basic_headers(styler):
    assert '{} & {A} & {B} & {C} \\\\' in styler.to_latex(siunitx=True)
    assert ' & A & B & C \\\\' in styler.to_latex()