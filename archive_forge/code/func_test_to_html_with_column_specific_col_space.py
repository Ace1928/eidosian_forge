from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_to_html_with_column_specific_col_space():
    df = DataFrame(np.random.default_rng(2).random(size=(3, 3)), columns=['a', 'b', 'c'])
    result = df.to_html(col_space={'a': '2em', 'b': 23})
    hdrs = [x for x in result.split('\n') if re.search('<th[>\\s]', x)]
    assert 'min-width: 2em;">a</th>' in hdrs[1]
    assert 'min-width: 23px;">b</th>' in hdrs[2]
    assert '<th>c</th>' in hdrs[3]
    result = df.to_html(col_space=['1em', 2, 3])
    hdrs = [x for x in result.split('\n') if re.search('<th[>\\s]', x)]
    assert 'min-width: 1em;">a</th>' in hdrs[1]
    assert 'min-width: 2px;">b</th>' in hdrs[2]
    assert 'min-width: 3px;">c</th>' in hdrs[3]