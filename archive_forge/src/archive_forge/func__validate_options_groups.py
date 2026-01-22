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
def _validate_options_groups(self, *events):
    if self.options and self.groups:
        raise ValueError(f'{type(self).__name__} options and groups parameters are mutually exclusive.')
    if self.size > 1 and self.groups:
        raise ValueError(f'{type(self).__name__} with size > 1 doe not support the `groups` parameter, use `options` instead.')