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
@param.depends('value', watch=True)
def _update_options_programmatically(self):
    """
        When value is passed, update to the latest options.
        """
    if self.options is None:
        return
    options = self.options if callable(self.options) else self.options.copy()
    set_values = self.value.copy()
    original_values = self._gather_values_from_widgets()
    if set_values == original_values:
        return
    with param.parameterized.batch_call_watchers(self):
        try:
            for i in range(self._max_depth):
                curr_select = self._widgets[i]
                if callable(options):
                    options = self._resolve_callable_options(i, options)
                    curr_options = list(options)
                elif isinstance(options, dict):
                    curr_options = list(options.keys())
                else:
                    curr_options = options
                curr_value = self._lookup_value(i, curr_options, set_values, name=curr_select.name, error=True)
                with param.discard_events(self):
                    curr_select.param.update(options=curr_options, value=curr_value, visible=callable(curr_options) or len(curr_options) > 0)
                if curr_value is None:
                    break
                if i < self._max_depth - 1:
                    options = options[curr_value]
        except Exception:
            self.value = original_values
            raise