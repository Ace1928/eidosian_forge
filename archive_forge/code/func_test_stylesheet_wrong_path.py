from __future__ import annotations
from io import (
import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.common import get_handle
from pandas.io.xml import read_xml
def test_stylesheet_wrong_path(geom_df):
    lxml_etree = pytest.importorskip('lxml.etree')
    xsl = os.path.join('data', 'xml', 'row_field_output.xslt')
    with pytest.raises(lxml_etree.XMLSyntaxError, match="Start tag expected, '<' not found"):
        geom_df.to_xml(stylesheet=xsl)