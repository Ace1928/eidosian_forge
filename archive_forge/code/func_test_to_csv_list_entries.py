from datetime import datetime
from io import StringIO
import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
from pandas.io.common import get_handle
def test_to_csv_list_entries(self):
    s = Series(['jack and jill', 'jesse and frank'])
    split = s.str.split('\\s+and\\s+')
    buf = StringIO()
    split.to_csv(buf, header=False)