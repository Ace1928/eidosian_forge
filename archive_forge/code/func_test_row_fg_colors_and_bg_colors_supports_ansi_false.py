import os
import pytest
from wasabi.tables import row, table
from wasabi.util import supports_ansi
def test_row_fg_colors_and_bg_colors_supports_ansi_false(fg_colors, bg_colors):
    os.environ['ANSI_COLORS_DISABLED'] = 'True'
    result = row(('Hello', 'World', '12344342'), fg_colors=fg_colors, bg_colors=bg_colors)
    assert result == 'Hello   World   12344342'
    del os.environ['ANSI_COLORS_DISABLED']