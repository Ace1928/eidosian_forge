from __future__ import annotations
import re
from typing import (
import warnings
from pandas.errors import CSSWarning
from pandas.util._exceptions import find_stack_level
def _update_font_size(self, props: dict[str, str], inherited: dict[str, str]) -> dict[str, str]:
    if props.get('font-size'):
        props['font-size'] = self.size_to_pt(props['font-size'], self._get_font_size(inherited), conversions=self.FONT_SIZE_RATIOS)
    return props