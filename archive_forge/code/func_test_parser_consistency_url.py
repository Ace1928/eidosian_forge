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
@pytest.mark.network
@pytest.mark.single_cpu
def test_parser_consistency_url(parser, httpserver):
    httpserver.serve_content(content=xml_default_nmsp)
    df_xpath = read_xml(StringIO(xml_default_nmsp), parser=parser)
    df_iter = read_xml(BytesIO(xml_default_nmsp.encode()), parser=parser, iterparse={'row': ['shape', 'degrees', 'sides']})
    tm.assert_frame_equal(df_xpath, df_iter)