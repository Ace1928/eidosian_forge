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
def _extract_level_metadata(self, i):
    """
        Extract the widget type and keyword arguments from the level metadata.
        """
    level = self._levels[i]
    if isinstance(level, int):
        return (Select, {})
    elif isinstance(level, str):
        return (Select, {'name': level})
    widget_type = level.get('type', Select)
    widget_kwargs = {k: v for k, v in level.items() if k != 'type'}
    return (widget_type, widget_kwargs)