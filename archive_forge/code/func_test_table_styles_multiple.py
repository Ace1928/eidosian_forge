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
def test_table_styles_multiple(self, df):
    ctx = df.style.set_table_styles([{'selector': 'th,td', 'props': 'color:red;'}, {'selector': 'tr', 'props': 'color:green;'}])._translate(True, True)['table_styles']
    assert ctx == [{'selector': 'th', 'props': [('color', 'red')]}, {'selector': 'td', 'props': [('color', 'red')]}, {'selector': 'tr', 'props': [('color', 'green')]}]