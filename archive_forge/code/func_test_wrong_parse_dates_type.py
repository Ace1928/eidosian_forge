from __future__ import annotations
from io import StringIO
import pytest
from pandas.errors import ParserWarning
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.xml import read_xml
def test_wrong_parse_dates_type(xml_books, parser, iterparse):
    with pytest.raises(TypeError, match='Only booleans, lists, and dictionaries are accepted'):
        read_xml(xml_books, parse_dates={'date'}, parser=parser, iterparse=iterparse)