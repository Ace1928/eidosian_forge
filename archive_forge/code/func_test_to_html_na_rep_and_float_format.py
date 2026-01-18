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
@pytest.mark.parametrize('na_rep', ['NaN', 'Ted'])
def test_to_html_na_rep_and_float_format(na_rep, datapath):
    df = DataFrame([['A', 1.2225], ['A', None]], columns=['Group', 'Data'])
    result = df.to_html(na_rep=na_rep, float_format='{:.2f}'.format)
    expected = expected_html(datapath, 'gh13828_expected_output')
    expected = expected.format(na_rep=na_rep)
    assert result == expected