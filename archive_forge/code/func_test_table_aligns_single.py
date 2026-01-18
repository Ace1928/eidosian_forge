import os
import pytest
from wasabi.tables import row, table
from wasabi.util import supports_ansi
def test_table_aligns_single(data):
    result = table(data, aligns='r')
    assert result == '\n         Hello   World   12344342\nThis is a test   World       1234\n'