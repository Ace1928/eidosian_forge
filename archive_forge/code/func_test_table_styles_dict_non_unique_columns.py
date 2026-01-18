from textwrap import dedent
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_table_styles_dict_non_unique_columns(styler):
    styles = styler.set_table_styles({'d': [{'selector': 'td', 'props': 'a: v;'}]}, axis=0).table_styles
    assert styles == [{'selector': 'td.col1', 'props': [('a', 'v')]}, {'selector': 'td.col2', 'props': [('a', 'v')]}]