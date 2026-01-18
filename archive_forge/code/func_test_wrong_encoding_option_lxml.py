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
@pytest.mark.parametrize('encoding', ['UTF-8', 'UTF-16', 'ISO-8859-1'])
def test_wrong_encoding_option_lxml(xml_baby_names, parser, encoding):
    pytest.importorskip('lxml')
    df_file = read_xml(xml_baby_names, encoding='ISO-8859-1', parser='lxml')
    with tm.ensure_clean('test.xml') as path:
        df_file.to_xml(path, index=False, encoding=encoding, parser=parser)