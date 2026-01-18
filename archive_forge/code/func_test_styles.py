from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_styles(styler):
    styler.set_uuid('abc')
    styler.set_table_styles([{'selector': 'td', 'props': 'color: red;'}])
    result = styler.to_html(doctype_html=True)
    expected = dedent('        <!DOCTYPE html>\n        <html>\n        <head>\n        <meta charset="utf-8">\n        <style type="text/css">\n        #T_abc td {\n          color: red;\n        }\n        </style>\n        </head>\n        <body>\n        <table id="T_abc">\n          <thead>\n            <tr>\n              <th class="blank level0" >&nbsp;</th>\n              <th id="T_abc_level0_col0" class="col_heading level0 col0" >A</th>\n            </tr>\n          </thead>\n          <tbody>\n            <tr>\n              <th id="T_abc_level0_row0" class="row_heading level0 row0" >a</th>\n              <td id="T_abc_row0_col0" class="data row0 col0" >2.610000</td>\n            </tr>\n            <tr>\n              <th id="T_abc_level0_row1" class="row_heading level0 row1" >b</th>\n              <td id="T_abc_row1_col0" class="data row1 col0" >2.690000</td>\n            </tr>\n          </tbody>\n        </table>\n        </body>\n        </html>\n        ')
    assert result == expected