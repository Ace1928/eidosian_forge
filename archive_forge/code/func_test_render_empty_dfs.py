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
def test_render_empty_dfs(self):
    empty_df = DataFrame()
    es = Styler(empty_df)
    es.to_html()
    DataFrame(columns=['a']).style.to_html()
    DataFrame(index=['a']).style.to_html()