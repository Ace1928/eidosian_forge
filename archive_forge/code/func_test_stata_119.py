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
@pytest.mark.slow
def test_stata_119(self, datapath):
    with gzip.open(datapath('io', 'data', 'stata', 'stata1_119.dta.gz'), 'rb') as gz:
        with StataReader(gz) as reader:
            reader._ensure_open()
            assert reader._nvar == 32999