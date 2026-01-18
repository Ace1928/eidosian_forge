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
def test_render_empty_mi():
    df = DataFrame(index=MultiIndex.from_product([['A'], [0, 1]], names=[None, 'one']))
    expected = dedent('    >\n      <thead>\n        <tr>\n          <th class="index_name level0" >&nbsp;</th>\n          <th class="index_name level1" >one</th>\n        </tr>\n      </thead>\n    ')
    assert expected in df.style.to_html()