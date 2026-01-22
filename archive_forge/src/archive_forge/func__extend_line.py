from __future__ import annotations
from collections.abc import (
import sys
from typing import (
from unicodedata import east_asian_width
from pandas._config import get_option
from pandas.core.dtypes.inference import is_sequence
from pandas.io.formats.console import get_console_size
def _extend_line(s: str, line: str, value: str, display_width: int, next_line_prefix: str) -> tuple[str, str]:
    if adj.len(line.rstrip()) + adj.len(value.rstrip()) >= display_width:
        s += line.rstrip()
        line = next_line_prefix
    line += value
    return (s, line)