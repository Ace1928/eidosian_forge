from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
@pytest.mark.parametrize('index', [False, True])
@pytest.mark.parametrize('columns', [False, True])
@pytest.mark.parametrize('index_name', [True, False])
def test_sticky_basic(styler, index, columns, index_name):
    if index_name:
        styler.index.name = 'some text'
    if index:
        styler.set_sticky(axis=0)
    if columns:
        styler.set_sticky(axis=1)
    left_css = '#T_ {0} {{\n  position: sticky;\n  background-color: inherit;\n  left: 0px;\n  z-index: {1};\n}}'
    top_css = '#T_ {0} {{\n  position: sticky;\n  background-color: inherit;\n  top: {1}px;\n  z-index: {2};\n{3}}}'
    res = styler.set_uuid('').to_html()
    assert (left_css.format('thead tr th:nth-child(1)', '3 !important') in res) is index
    assert (left_css.format('tbody tr th:nth-child(1)', '1') in res) is index
    assert (top_css.format('thead tr:nth-child(1) th', '0', '2', '  height: 25px;\n') in res) is (columns and index_name)
    assert (top_css.format('thead tr:nth-child(2) th', '25', '2', '  height: 25px;\n') in res) is (columns and index_name)
    assert (top_css.format('thead tr:nth-child(1) th', '0', '2', '') in res) is (columns and (not index_name))