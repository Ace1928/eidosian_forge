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
def test_unicode_dta_118(self, datapath):
    unicode_df = self.read_dta(datapath('io', 'data', 'stata', 'stata16_118.dta'))
    columns = ['utf8', 'latin1', 'ascii', 'utf8_strl', 'ascii_strl']
    values = [['ραηδας', 'PÄNDÄS', 'p', 'ραηδας', 'p'], ['ƤĀńĐąŜ', 'Ö', 'a', 'ƤĀńĐąŜ', 'a'], ['ᴘᴀᴎᴅᴀS', 'Ü', 'n', 'ᴘᴀᴎᴅᴀS', 'n'], ['      ', '      ', 'd', '      ', 'd'], [' ', '', 'a', ' ', 'a'], ['', '', 's', '', 's'], ['', '', ' ', '', ' ']]
    expected = DataFrame(values, columns=columns)
    tm.assert_frame_equal(unicode_df, expected)