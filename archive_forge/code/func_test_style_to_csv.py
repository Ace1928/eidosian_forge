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
def test_style_to_csv(geom_df):
    pytest.importorskip('lxml')
    xsl = '<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">\n    <xsl:output method="text" indent="yes" />\n    <xsl:strip-space elements="*"/>\n\n    <xsl:param name="delim">,</xsl:param>\n    <xsl:template match="/data">\n        <xsl:text>,shape,degrees,sides&#xa;</xsl:text>\n        <xsl:apply-templates select="row"/>\n    </xsl:template>\n\n    <xsl:template match="row">\n        <xsl:value-of select="concat(index, $delim, shape, $delim,\n                                     degrees, $delim, sides)"/>\n         <xsl:text>&#xa;</xsl:text>\n    </xsl:template>\n</xsl:stylesheet>'
    out_csv = geom_df.to_csv(lineterminator='\n')
    if out_csv is not None:
        out_csv = out_csv.strip()
    out_xml = geom_df.to_xml(stylesheet=xsl)
    assert out_csv == out_xml