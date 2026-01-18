import os
import pytest
from wasabi.tables import row, table
from wasabi.util import supports_ansi
def test_table_header(data, header):
    result = table(data, header=header)
    assert result == '\nCOL A            COL B   COL 3   \nHello            World   12344342\nThis is a test   World   1234    \n'