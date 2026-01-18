import bz2
import datetime as dt
from datetime import datetime
import gzip
import io
import os
import struct
import tarfile
import zipfile
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import CategoricalDtype
import pandas._testing as tm
from pandas.core.frame import (
from pandas.io.parsers import read_csv
from pandas.io.stata import (
@pytest.mark.parametrize('version', [114, 117, 118, 119, None])
def test_timestamp_and_label(self, version):
    original = DataFrame([(1,)], columns=['variable'])
    time_stamp = datetime(2000, 2, 29, 14, 21)
    data_label = 'This is a data file.'
    with tm.ensure_clean() as path:
        original.to_stata(path, time_stamp=time_stamp, data_label=data_label, version=version)
        with StataReader(path) as reader:
            assert reader.time_stamp == '29 Feb 2000 14:21'
            assert reader.data_label == data_label