import os
import pytest
from wasabi.tables import row, table
from wasabi.util import supports_ansi
def test_table_widths():
    data = [('a', 'bb', 'ccc'), ('d', 'ee', 'fff')]
    widths = (5, 2, 10)
    result = table(data, widths=widths)
    assert result == '\na       bb   ccc       \nd       ee   fff       \n'