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
class _RadioGroupBase(SingleSelectBase):
    _supports_embed = False
    _rename: ClassVar[Mapping[str, str | None]] = {'name': None, 'options': 'labels', 'value': 'active'}
    _source_transforms = {'value': 'source.labels[value]'}
    _target_transforms = {'value': 'target.labels.indexOf(value)'}
    __abstract = True

    def _process_param_change(self, msg):
        msg = super(SingleSelectBase, self)._process_param_change(msg)
        values = self.values
        if 'active' in msg:
            value = msg['active']
            if value in values:
                msg['active'] = indexOf(value, values)
            else:
                if self.value is not None:
                    self.value = None
                msg['active'] = None
        if 'labels' in msg:
            msg['labels'] = self.labels
            value = self.value
            if not isIn(value, values):
                self.value = None
        return msg

    def _process_property_change(self, msg):
        msg = super(SingleSelectBase, self)._process_property_change(msg)
        if 'value' in msg:
            index = msg['value']
            if index is None:
                msg['value'] = None
            else:
                msg['value'] = list(self.values)[index]
        return msg

    def _get_embed_state(self, root, values=None, max_opts=3):
        if values is None:
            values = self.values
        elif any((v not in self.values for v in values)):
            raise ValueError('Supplied embed states were not found in the %s widgets values list.' % type(self).__name__)
        return (self, self._models[root.ref['id']][0], values, lambda x: x.active, 'active', 'cb_obj.active')