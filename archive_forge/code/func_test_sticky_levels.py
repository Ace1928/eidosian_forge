from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
@pytest.mark.parametrize('index', [False, True])
@pytest.mark.parametrize('columns', [False, True])
@pytest.mark.parametrize('levels', [[1], ['one'], 'one'])
def test_sticky_levels(styler_mi, index, columns, levels):
    styler_mi.index.names, styler_mi.columns.names = (['zero', 'one'], ['zero', 'one'])
    if index:
        styler_mi.set_sticky(axis=0, levels=levels)
    if columns:
        styler_mi.set_sticky(axis=1, levels=levels)
    left_css = '#T_ {0} {{\n  position: sticky;\n  background-color: inherit;\n  left: {1}px;\n  min-width: 75px;\n  max-width: 75px;\n  z-index: {2};\n}}'
    top_css = '#T_ {0} {{\n  position: sticky;\n  background-color: inherit;\n  top: {1}px;\n  height: 25px;\n  z-index: {2};\n}}'
    res = styler_mi.set_uuid('').to_html()
    assert '#T_ thead tr th:nth-child(1)' not in res
    assert '#T_ tbody tr th.level0' not in res
    assert '#T_ thead tr:nth-child(1) th' not in res
    assert (left_css.format('thead tr th:nth-child(2)', '0', '3 !important') in res) is index
    assert (left_css.format('tbody tr th.level1', '0', '1') in res) is index
    assert (top_css.format('thead tr:nth-child(2) th', '0', '2') in res) is columns