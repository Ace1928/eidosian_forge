import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
def test_str_escape_error():
    msg = "`escape` only permitted in {'html', 'latex', 'latex-math'}, got "
    with pytest.raises(ValueError, match=msg):
        _str_escape('text', 'bad_escape')
    with pytest.raises(ValueError, match=msg):
        _str_escape('text', [])
    _str_escape(2.0, 'bad_escape')