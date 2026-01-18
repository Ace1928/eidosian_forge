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
def test_to_html_encoding(float_frame, tmp_path):
    path = tmp_path / 'test.html'
    float_frame.to_html(path, encoding='gbk')
    with open(str(path), encoding='gbk') as f:
        assert float_frame.to_html() == f.read()