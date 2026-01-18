import string
import pytest
from pandas.errors import CSSWarning
import pandas._testing as tm
from pandas.io.formats.excel import (
def tests_css_named_colors_valid():
    upper_hexs = set(map(str.upper, string.hexdigits))
    for color in CSSToExcelConverter.NAMED_COLORS.values():
        assert len(color) == 6 and all((c in upper_hexs for c in color))