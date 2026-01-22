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
class _CheckGroupBase(SingleSelectBase):
    value = param.List(default=[])
    _rename: ClassVar[Mapping[str, str | None]] = {'name': None, 'options': 'labels', 'value': 'active'}
    _source_transforms = {'value': 'value.map((index) => source.labels[index])'}
    _target_transforms = {'value': 'value.map((label) => target.labels.indexOf(label))'}
    _supports_embed = False
    __abstract = True

    def _process_param_change(self, msg):
        msg = super()._process_param_change(msg)
        values = self.values
        if 'active' in msg:
            msg['active'] = [indexOf(v, values) for v in msg['active'] if isIn(v, values)]
        if 'labels' in msg:
            msg['labels'] = self.labels
            if any((not isIn(v, values) for v in self.value)):
                self.value = [v for v in self.value if isIn(v, values)]
            msg['active'] = [indexOf(v, values) for v in self.value if isIn(v, values)]
        msg.pop('title', None)
        return msg

    def _process_property_change(self, msg):
        msg = super(SingleSelectBase, self)._process_property_change(msg)
        if 'value' in msg:
            values = self.values
            msg['value'] = [values[a] for a in msg['value']]
        return msg