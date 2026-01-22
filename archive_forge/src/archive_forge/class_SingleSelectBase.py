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
class SingleSelectBase(SelectBase):
    value = param.Parameter(default=None)
    _supports_embed: ClassVar[bool] = True
    __abstract = True

    def __init__(self, **params):
        super().__init__(**params)
        values = self.values
        if self.value is None and None not in values and values:
            self.value = values[0]

    def _process_param_change(self, msg):
        msg = super()._process_param_change(msg)
        labels, values = (self.labels, self.values)
        unique = len(set(self.unicode_values)) == len(labels)
        if 'value' in msg:
            val = msg['value']
            if isIn(val, values):
                unicode_values = self.unicode_values if unique else labels
                msg['value'] = unicode_values[indexOf(val, values)]
            elif values:
                self.value = self.values[0]
            else:
                self.value = None
                msg['value'] = ''
        if 'options' in msg:
            if isinstance(self.options, dict):
                if unique:
                    options = [(v, l) for l, v in zip(labels, self.unicode_values)]
                else:
                    options = labels
                msg['options'] = options
            else:
                msg['options'] = self.unicode_values
            val = self.value
            if values:
                if not isIn(val, values):
                    self.value = values[0]
            else:
                self.value = None
        return msg

    @property
    def unicode_values(self):
        return [str(v) for v in self.values]

    def _process_property_change(self, msg):
        msg = super()._process_property_change(msg)
        if 'value' in msg:
            if not self.values:
                pass
            elif msg['value'] == '':
                msg['value'] = self.values[0] if self.values else None
            else:
                if isIn(msg['value'], self.unicode_values):
                    idx = indexOf(msg['value'], self.unicode_values)
                else:
                    idx = indexOf(msg['value'], self.labels)
                msg['value'] = self._items[self.labels[idx]]
        msg.pop('options', None)
        return msg

    def _get_embed_state(self, root, values=None, max_opts=3):
        if values is None:
            values = self.values
        elif any((v not in self.values for v in values)):
            raise ValueError('Supplied embed states were not found in the %s widgets values list.' % type(self).__name__)
        return (self, self._models[root.ref['id']][0], values, lambda x: x.value, 'value', 'cb_obj.value')