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
def test_write_variable_label_errors(self, mixed_frame):
    values = ['Ρ', 'Α', 'Ν', 'Δ', 'Α', 'Σ']
    variable_labels_utf8 = {'a': 'City Rank', 'b': 'City Exponent', 'c': ''.join(values)}
    msg = 'Variable labels must contain only characters that can be encoded in Latin-1'
    with pytest.raises(ValueError, match=msg):
        with tm.ensure_clean() as path:
            mixed_frame.to_stata(path, variable_labels=variable_labels_utf8)
    variable_labels_long = {'a': 'City Rank', 'b': 'City Exponent', 'c': 'A very, very, very long variable label that is too long for Stata which means that it has more than 80 characters'}
    msg = 'Variable labels must be 80 characters or fewer'
    with pytest.raises(ValueError, match=msg):
        with tm.ensure_clean() as path:
            mixed_frame.to_stata(path, variable_labels=variable_labels_long)