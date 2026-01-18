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
def test_to_html_justify(justify, datapath):
    df = DataFrame({'A': [6, 30000, 2], 'B': [1, 2, 70000], 'C': [223442, 0, 1]}, columns=['A', 'B', 'C'])
    result = df.to_html(justify=justify)
    expected = expected_html(datapath, 'justify').format(justify=justify)
    assert result == expected