from __future__ import annotations
from collections.abc import (
import functools
import itertools
import re
from typing import (
import warnings
import numpy as np
from pandas._libs.lib import is_list_like
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes import missing
from pandas.core.dtypes.common import (
from pandas import (
import pandas.core.common as com
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats._color_data import CSS4_COLORS
from pandas.io.formats.css import (
from pandas.io.formats.format import get_level_lengths
from pandas.io.formats.printing import pprint_thing
def _border_style(self, style: str | None, width: str | None, color: str | None):
    if width is None and style is None and (color is None):
        return None
    if width is None and style is None:
        return 'none'
    if style in ('none', 'hidden'):
        return 'none'
    width_name = self._get_width_name(width)
    if width_name is None:
        return 'none'
    if style in (None, 'groove', 'ridge', 'inset', 'outset', 'solid'):
        return width_name
    if style == 'double':
        return 'double'
    if style == 'dotted':
        if width_name in ('hair', 'thin'):
            return 'dotted'
        return 'mediumDashDotDot'
    if style == 'dashed':
        if width_name in ('hair', 'thin'):
            return 'dashed'
        return 'mediumDashed'
    elif style in self.BORDER_STYLE_MAP:
        return self.BORDER_STYLE_MAP[style]
    else:
        warnings.warn(f'Unhandled border style format: {repr(style)}', CSSWarning, stacklevel=find_stack_level())
        return 'none'