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
def test_table_attributes(self, df):
    attributes = 'class="foo" data-bar'
    styler = Styler(df, table_attributes=attributes)
    result = styler.to_html()
    assert 'class="foo" data-bar' in result
    result = df.style.set_table_attributes(attributes).to_html()
    assert 'class="foo" data-bar' in result