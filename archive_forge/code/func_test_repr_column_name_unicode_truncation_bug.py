from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_repr_column_name_unicode_truncation_bug(self):
    df = DataFrame({'Id': [7117434], 'StringCol': 'Is it possible to modify drop plot codeso that the output graph is displayed in iphone simulator, Is it possible to modify drop plot code so that the output graph is â\x80¨displayed in iphone simulator.Now we are adding the CSV file externally. I want to Call the File through the code..'})
    with option_context('display.max_columns', 20):
        assert 'StringCol' in repr(df)