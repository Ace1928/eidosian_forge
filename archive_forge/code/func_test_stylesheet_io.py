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
def test_stylesheet_io(xsl_row_field_output, mode, geom_df):
    pytest.importorskip('lxml')
    xsl_obj: BytesIO | StringIO
    with open(xsl_row_field_output, mode, encoding='utf-8' if mode == 'r' else None) as f:
        if mode == 'rb':
            xsl_obj = BytesIO(f.read())
        else:
            xsl_obj = StringIO(f.read())
    output = geom_df.to_xml(stylesheet=xsl_obj)
    assert output == xsl_expected