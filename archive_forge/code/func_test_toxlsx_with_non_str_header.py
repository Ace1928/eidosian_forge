from __future__ import absolute_import, print_function, division
from datetime import datetime
from tempfile import NamedTemporaryFile
import pytest
import petl as etl
from petl.io.xlsx import fromxlsx, toxlsx, appendxlsx
from petl.test.helpers import ieq, eq_
def test_toxlsx_with_non_str_header(xlsx_table_with_non_str_header):
    f = NamedTemporaryFile(delete=True, suffix='.xlsx')
    f.close()
    toxlsx(xlsx_table_with_non_str_header, f.name, 'Sheet1')
    actual = etl.fromxlsx(f.name, 'Sheet1')
    ieq(xlsx_table_with_non_str_header, actual)