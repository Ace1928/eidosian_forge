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
@pytest.mark.parametrize('kwargs,string,expected', [({}, "<type 'str'>", 'escaped'), ({'escape': False}, '<b>bold</b>', 'escape_disabled')])
def test_to_html_escaped(kwargs, string, expected, datapath):
    a = 'str<ing1 &amp;'
    b = 'stri>ng2 &amp;'
    test_dict = {'co<l1': {a: string, b: string}, 'co>l2': {a: string, b: string}}
    result = DataFrame(test_dict).to_html(**kwargs)
    expected = expected_html(datapath, expected)
    assert result == expected