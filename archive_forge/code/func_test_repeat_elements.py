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
def test_repeat_elements(parser):
    xml = '<shapes>\n  <shape>\n    <value item="name">circle</value>\n    <value item="family">ellipse</value>\n    <value item="degrees">360</value>\n    <value item="sides">0</value>\n  </shape>\n  <shape>\n    <value item="name">triangle</value>\n    <value item="family">polygon</value>\n    <value item="degrees">180</value>\n    <value item="sides">3</value>\n  </shape>\n  <shape>\n    <value item="name">square</value>\n    <value item="family">polygon</value>\n    <value item="degrees">360</value>\n    <value item="sides">4</value>\n  </shape>\n</shapes>'
    df_xpath = read_xml(StringIO(xml), xpath='.//shape', parser=parser, names=['name', 'family', 'degrees', 'sides'])
    df_iter = read_xml_iterparse(xml, parser=parser, iterparse={'shape': ['value', 'value', 'value', 'value']}, names=['name', 'family', 'degrees', 'sides'])
    df_expected = DataFrame({'name': ['circle', 'triangle', 'square'], 'family': ['ellipse', 'polygon', 'polygon'], 'degrees': [360, 180, 360], 'sides': [0, 3, 4]})
    tm.assert_frame_equal(df_xpath, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)