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
@pytest.mark.parametrize('option, val', [('styler.render.max_elements', 6), ('styler.render.max_rows', 3)])
def test_render_trimming_rows(option, val):
    df = DataFrame(np.arange(120).reshape(60, 2))
    with option_context(option, val):
        ctx = df.style._translate(True, True)
    assert len(ctx['head'][0]) == 3
    assert len(ctx['body']) == 4
    assert len(ctx['body'][0]) == 3