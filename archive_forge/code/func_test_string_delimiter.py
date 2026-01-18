from textwrap import dedent
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_string_delimiter(styler):
    result = styler.to_string(delimiter=';')
    expected = dedent('    ;A;B;C\n    0;0;-0.61;ab\n    1;1;-1.22;cd\n    ')
    assert result == expected