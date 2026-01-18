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
def test_to_html_truncate(datapath):
    index = pd.date_range(start='20010101', freq='D', periods=20)
    df = DataFrame(index=index, columns=range(20))
    result = df.to_html(max_rows=8, max_cols=4)
    expected = expected_html(datapath, 'truncate')
    assert result == expected