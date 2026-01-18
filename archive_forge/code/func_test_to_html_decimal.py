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
def test_to_html_decimal(datapath):
    df = DataFrame({'A': [6.0, 3.1, 2.2]})
    result = df.to_html(decimal=',')
    expected = expected_html(datapath, 'gh12031_expected_output')
    assert result == expected