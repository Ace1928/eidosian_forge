from a list of options.
from __future__ import annotations
import itertools
import re
from types import FunctionType
from typing import (
import numpy as np
import param
from bokeh.models import PaletteSelect
from bokeh.models.widgets import (
from ..io.resources import CDN_DIST
from ..layout.base import Column, ListPanel, NamedListPanel
from ..models import (
from ..util import PARAM_NAME_PATTERN, indexOf, isIn
from ._mixin import TooltipMixin
from .base import CompositeWidget, Widget
from .button import Button, _ButtonBase
from .input import TextAreaInput, TextInput
def _find_max_depth(self, d, depth=1):
    if d is None or len(d) == 0:
        return 0
    elif not isinstance(d, dict):
        return depth
    max_depth = depth
    for value in d.values():
        if isinstance(value, dict):
            max_depth = max(max_depth, self._find_max_depth(value, depth + 1))
        if isinstance(value, list) and len(value) == 0 and (max_depth > 0):
            max_depth -= 1
    return max_depth