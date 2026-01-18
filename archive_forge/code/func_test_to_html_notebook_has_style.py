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
@pytest.mark.parametrize('notebook', [True, False])
def test_to_html_notebook_has_style(notebook):
    df = DataFrame({'A': [1, 2, 3]})
    result = df.to_html(notebook=notebook)
    if notebook:
        assert 'tbody tr th:only-of-type' in result
        assert 'vertical-align: middle;' in result
        assert 'thead th' in result
    else:
        assert 'tbody tr th:only-of-type' not in result
        assert 'vertical-align: middle;' not in result
        assert 'thead th' not in result