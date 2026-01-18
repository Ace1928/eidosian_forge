import io
import os
import sys
from zipfile import ZipFile
from _csv import Error
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('errors', ['surrogatepass', 'ignore', 'replace'])
def test_to_csv_errors(self, errors):
    data = ['\ud800foo']
    ser = pd.Series(data, index=Index(data, dtype=object), dtype=object)
    with tm.ensure_clean('test.csv') as path:
        ser.to_csv(path, errors=errors)