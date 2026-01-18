import os
import pytest
from wasabi.tables import row, table
from wasabi.util import supports_ansi
def test_table_header_footer_divider(data, header, footer):
    result = table(data, header=header, footer=footer, divider=True)
    assert result == '\nCOL A            COL B   COL 3     \n--------------   -----   ----------\nHello            World   12344342  \nThis is a test   World   1234      \n--------------   -----   ----------\n                         2030203.00\n'