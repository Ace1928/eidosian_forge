import numpy as np
import pytest
from pandas import DataFrame
from pandas.io.formats.style import Styler
def test_tooltip_ignored(styler):
    result = styler.to_html()
    assert '<style type="text/css">\n</style>' in result
    assert '<span class="pd-t"></span>' not in result