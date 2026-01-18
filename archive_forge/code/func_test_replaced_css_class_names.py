from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_replaced_css_class_names():
    css = {'row_heading': 'ROWHEAD', 'index_name': 'IDXNAME', 'row': 'ROW', 'row_trim': 'ROWTRIM', 'level': 'LEVEL', 'data': 'DATA', 'blank': 'BLANK'}
    midx = MultiIndex.from_product([['a', 'b'], ['c', 'd']])
    styler_mi = Styler(DataFrame(np.arange(16).reshape(4, 4), index=midx, columns=midx), uuid_len=0).set_table_styles(css_class_names=css)
    styler_mi.index.names = ['n1', 'n2']
    styler_mi.hide(styler_mi.index[1:], axis=0)
    styler_mi.hide(styler_mi.columns[1:], axis=1)
    styler_mi.map_index(lambda v: 'color: red;', axis=0)
    styler_mi.map_index(lambda v: 'color: green;', axis=1)
    styler_mi.map(lambda v: 'color: blue;')
    expected = dedent('    <style type="text/css">\n    #T__ROW0_col0 {\n      color: blue;\n    }\n    #T__LEVEL0_ROW0, #T__LEVEL1_ROW0 {\n      color: red;\n    }\n    #T__LEVEL0_col0, #T__LEVEL1_col0 {\n      color: green;\n    }\n    </style>\n    <table id="T_">\n      <thead>\n        <tr>\n          <th class="BLANK" >&nbsp;</th>\n          <th class="IDXNAME LEVEL0" >n1</th>\n          <th id="T__LEVEL0_col0" class="col_heading LEVEL0 col0" >a</th>\n        </tr>\n        <tr>\n          <th class="BLANK" >&nbsp;</th>\n          <th class="IDXNAME LEVEL1" >n2</th>\n          <th id="T__LEVEL1_col0" class="col_heading LEVEL1 col0" >c</th>\n        </tr>\n        <tr>\n          <th class="IDXNAME LEVEL0" >n1</th>\n          <th class="IDXNAME LEVEL1" >n2</th>\n          <th class="BLANK col0" >&nbsp;</th>\n        </tr>\n      </thead>\n      <tbody>\n        <tr>\n          <th id="T__LEVEL0_ROW0" class="ROWHEAD LEVEL0 ROW0" >a</th>\n          <th id="T__LEVEL1_ROW0" class="ROWHEAD LEVEL1 ROW0" >c</th>\n          <td id="T__ROW0_col0" class="DATA ROW0 col0" >0</td>\n        </tr>\n      </tbody>\n    </table>\n    ')
    result = styler_mi.to_html()
    assert result == expected