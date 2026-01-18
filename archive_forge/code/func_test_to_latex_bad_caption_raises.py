import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('bad_caption', [('full_caption', 'short_caption', 'extra_string'), ('full_caption', 'short_caption', 1), ('full_caption', 'short_caption', None), ('full_caption',), (None,)])
def test_to_latex_bad_caption_raises(self, bad_caption):
    df = DataFrame({'a': [1]})
    msg = '`caption` must be either a string or 2-tuple of strings'
    with pytest.raises(ValueError, match=msg):
        df.to_latex(caption=bad_caption)