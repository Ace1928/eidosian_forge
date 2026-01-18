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
def test_repeated_column_labels(self, datapath):
    msg = '\nValue labels for column ethnicsn are not unique. These cannot be converted to\npandas categoricals.\n\nEither read the file with `convert_categoricals` set to False or use the\nlow level interface in `StataReader` to separately read the values and the\nvalue_labels.\n\nThe repeated labels are:\n-+\nwolof\n'
    with pytest.raises(ValueError, match=msg):
        read_stata(datapath('io', 'data', 'stata', 'stata15.dta'), convert_categoricals=True)