import contextlib
import copy
import re
from textwrap import dedent
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.formats.style import (  # isort:skip
from pandas.io.formats.style_render import (
def test_render_double(self):
    df = DataFrame({'A': [0, 1]})
    style = lambda x: Series(['color: red; border: 1px', 'color: blue; border: 2px'], name=x.name)
    s = Styler(df, uuid='AB').apply(style)
    s.to_html()