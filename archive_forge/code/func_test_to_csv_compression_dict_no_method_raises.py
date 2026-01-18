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
def test_to_csv_compression_dict_no_method_raises(self):
    df = DataFrame({'ABC': [1]})
    compression = {'some_option': True}
    msg = "must have key 'method'"
    with tm.ensure_clean('out.zip') as path:
        with pytest.raises(ValueError, match=msg):
            df.to_csv(path, compression=compression)