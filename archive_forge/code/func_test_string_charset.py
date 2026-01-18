from __future__ import annotations
from io import (
from lzma import LZMAError
import os
from tarfile import ReadError
from urllib.error import HTTPError
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile
import numpy as np
import pytest
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.common import get_handle
from pandas.io.xml import read_xml
def test_string_charset(parser):
    txt = '<中文標籤><row><c1>1</c1><c2>2</c2></row></中文標籤>'
    df_str = read_xml(StringIO(txt), parser=parser)
    df_expected = DataFrame({'c1': 1, 'c2': 2}, index=[0])
    tm.assert_frame_equal(df_str, df_expected)