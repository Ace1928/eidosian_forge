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
def _uses_callable(self, d):
    """
        Check if the nested options has a callable.
        """
    if callable(d):
        return True
    if isinstance(d, dict):
        for value in d.values():
            if callable(value):
                return True
            elif isinstance(value, dict):
                return self._uses_callable(value)
    return False