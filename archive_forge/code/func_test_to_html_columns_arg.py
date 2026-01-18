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
def test_to_html_columns_arg(float_frame):
    result = float_frame.to_html(columns=['A'])
    assert '<th>B</th>' not in result