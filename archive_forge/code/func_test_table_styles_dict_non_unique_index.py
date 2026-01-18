from textwrap import dedent
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_table_styles_dict_non_unique_index(styler):
    styles = styler.set_table_styles({'j': [{'selector': 'td', 'props': 'a: v;'}]}, axis=1).table_styles
    assert styles == [{'selector': 'td.row1', 'props': [('a', 'v')]}, {'selector': 'td.row2', 'props': [('a', 'v')]}]