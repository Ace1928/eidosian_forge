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
def test_variable_labels(self, datapath):
    with StataReader(datapath('io', 'data', 'stata', 'stata7_115.dta')) as rdr:
        sr_115 = rdr.variable_labels()
    with StataReader(datapath('io', 'data', 'stata', 'stata7_117.dta')) as rdr:
        sr_117 = rdr.variable_labels()
    keys = ('var1', 'var2', 'var3')
    labels = ('label1', 'label2', 'label3')
    for k, v in sr_115.items():
        assert k in sr_117
        assert v == sr_117[k]
        assert k in keys
        assert v in labels