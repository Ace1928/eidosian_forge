import os
import pytest
from wasabi.tables import row, table
from wasabi.util import supports_ansi
def test_table_multiline(header):
    data = [('hello', ['foo', 'bar', 'baz'], 'world'), ('hello', 'world', ['world 1', 'world 2'])]
    result = table(data, header=header, divider=True, multiline=True)
    assert result == '\nCOL A   COL B   COL 3  \n-----   -----   -------\nhello   foo     world  \n        bar            \n        baz            \n                       \nhello   world   world 1\n                world 2\n'