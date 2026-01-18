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
def test_stylesheet_file(kml_cta_rail_lines, xsl_flatten_doc):
    pytest.importorskip('lxml')
    df_style = read_xml(kml_cta_rail_lines, xpath='.//k:Placemark', namespaces={'k': 'http://www.opengis.net/kml/2.2'}, stylesheet=xsl_flatten_doc)
    df_iter = read_xml(kml_cta_rail_lines, iterparse={'Placemark': ['id', 'name', 'styleUrl', 'extrude', 'altitudeMode', 'coordinates']})
    tm.assert_frame_equal(df_kml, df_style)
    tm.assert_frame_equal(df_kml, df_iter)