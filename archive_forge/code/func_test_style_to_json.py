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
def test_style_to_json(geom_df):
    pytest.importorskip('lxml')
    xsl = '<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">\n    <xsl:output method="text" indent="yes" />\n    <xsl:strip-space elements="*"/>\n\n    <xsl:param name="quot">"</xsl:param>\n\n    <xsl:template match="/data">\n        <xsl:text>{"shape":{</xsl:text>\n        <xsl:apply-templates select="descendant::row/shape"/>\n        <xsl:text>},"degrees":{</xsl:text>\n        <xsl:apply-templates select="descendant::row/degrees"/>\n        <xsl:text>},"sides":{</xsl:text>\n        <xsl:apply-templates select="descendant::row/sides"/>\n        <xsl:text>}}</xsl:text>\n    </xsl:template>\n\n    <xsl:template match="shape|degrees|sides">\n        <xsl:variable name="val">\n            <xsl:if test = ".=\'\'">\n                <xsl:value-of select="\'null\'"/>\n            </xsl:if>\n            <xsl:if test = "number(text()) = text()">\n                <xsl:value-of select="text()"/>\n            </xsl:if>\n            <xsl:if test = "number(text()) != text()">\n                <xsl:value-of select="concat($quot, text(), $quot)"/>\n            </xsl:if>\n        </xsl:variable>\n        <xsl:value-of select="concat($quot, preceding-sibling::index,\n                                     $quot,\':\', $val)"/>\n        <xsl:if test="preceding-sibling::index != //row[last()]/index">\n            <xsl:text>,</xsl:text>\n        </xsl:if>\n    </xsl:template>\n</xsl:stylesheet>'
    out_json = geom_df.to_json()
    out_xml = geom_df.to_xml(stylesheet=xsl)
    assert out_json == out_xml