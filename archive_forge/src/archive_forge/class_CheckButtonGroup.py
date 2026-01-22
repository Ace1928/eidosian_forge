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
class CheckButtonGroup(_CheckGroupBase, _ButtonBase, TooltipMixin):
    """
    The `CheckButtonGroup` widget allows selecting between a list of options
    by toggling the corresponding buttons.

    It falls into the broad category of multi-option selection widgets that
    provide a compatible API and include the `MultiSelect`, `CrossSelector`
    and `CheckBoxGroup` widgets.

    Reference: https://panel.holoviz.org/reference/widgets/CheckButtonGroup.html

    :Example:

    >>> CheckButtonGroup(
    ...     name='Regression Models', value=['Lasso', 'Ridge'],
    ...     options=['Lasso', 'Linear', 'Ridge', 'Polynomial']
    ... )
    """
    orientation = param.Selector(default='horizontal', objects=['horizontal', 'vertical'], doc="\n        Button group orientation, either 'horizontal' (default) or 'vertical'.")
    _rename: ClassVar[Mapping[str, str | None]] = {**_CheckGroupBase._rename, **TooltipMixin._rename}
    _source_transforms = {'value': 'value.map((index) => source.labels[index])', 'button_style': None, 'description': None}
    _widget_type: ClassVar[Type[Model]] = _BkCheckboxButtonGroup