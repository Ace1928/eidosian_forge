from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_rendered_links():
    df = DataFrame(['text www.domain.com text'])
    result = df.style.format(hyperlinks='latex').to_latex()
    assert 'text \\href{www.domain.com}{www.domain.com} text' in result