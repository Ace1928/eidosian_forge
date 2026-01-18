from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_parse_latex_table_styles(styler):
    styler.set_table_styles([{'selector': 'foo', 'props': [('attr', 'value')]}, {'selector': 'bar', 'props': [('attr', 'overwritten')]}, {'selector': 'bar', 'props': [('attr', 'baz'), ('attr2', 'ignored')]}, {'selector': 'label', 'props': [('', '{fig§item}')]}])
    assert _parse_latex_table_styles(styler.table_styles, 'bar') == 'baz'
    assert _parse_latex_table_styles(styler.table_styles, 'label') == '{fig:item}'