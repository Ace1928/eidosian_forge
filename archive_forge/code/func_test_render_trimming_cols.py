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
@pytest.mark.parametrize('option, val', [('styler.render.max_elements', 6), ('styler.render.max_columns', 2)])
def test_render_trimming_cols(option, val):
    df = DataFrame(np.arange(30).reshape(3, 10))
    with option_context(option, val):
        ctx = df.style._translate(True, True)
    assert len(ctx['head'][0]) == 4
    assert len(ctx['body']) == 3
    assert len(ctx['body'][0]) == 4