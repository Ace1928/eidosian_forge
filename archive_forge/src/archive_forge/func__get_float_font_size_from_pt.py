from __future__ import annotations
import re
from typing import (
import warnings
from pandas.errors import CSSWarning
from pandas.util._exceptions import find_stack_level
def _get_float_font_size_from_pt(self, font_size_string: str) -> float:
    assert font_size_string.endswith('pt')
    return float(font_size_string.rstrip('pt'))