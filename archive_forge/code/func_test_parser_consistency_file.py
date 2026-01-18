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
def test_parser_consistency_file(xml_books):
    pytest.importorskip('lxml')
    df_file_lxml = read_xml(xml_books, parser='lxml')
    df_file_etree = read_xml(xml_books, parser='etree')
    df_iter_lxml = read_xml(xml_books, parser='lxml', iterparse={'book': ['category', 'title', 'year', 'author', 'price']})
    df_iter_etree = read_xml(xml_books, parser='etree', iterparse={'book': ['category', 'title', 'year', 'author', 'price']})
    tm.assert_frame_equal(df_file_lxml, df_file_etree)
    tm.assert_frame_equal(df_file_lxml, df_iter_lxml)
    tm.assert_frame_equal(df_iter_lxml, df_iter_etree)