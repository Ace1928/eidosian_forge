import os
import pytest
from wasabi.tables import row, table
from wasabi.util import supports_ansi
def test_row_fg_colors_and_bg_colors_log_friendly(fg_colors, bg_colors):
    ENV_LOG_FRIENDLY = 'WASABI_LOG_FRIENDLY'
    os.environ[ENV_LOG_FRIENDLY] = 'True'
    result = row(('Hello', 'World', '12344342'), fg_colors=fg_colors, bg_colors=bg_colors)
    assert result == 'Hello   World   12344342'
    del os.environ[ENV_LOG_FRIENDLY]