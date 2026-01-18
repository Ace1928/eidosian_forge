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
def test_xml_declaration_pretty_print(geom_df):
    pytest.importorskip('lxml')
    expected = '<data>\n  <row>\n    <index>0</index>\n    <shape>square</shape>\n    <degrees>360</degrees>\n    <sides>4.0</sides>\n  </row>\n  <row>\n    <index>1</index>\n    <shape>circle</shape>\n    <degrees>360</degrees>\n    <sides/>\n  </row>\n  <row>\n    <index>2</index>\n    <shape>triangle</shape>\n    <degrees>180</degrees>\n    <sides>3.0</sides>\n  </row>\n</data>'
    output = geom_df.to_xml(xml_declaration=False)
    assert output == expected