from __future__ import annotations
import ast
import json
from base64 import b64decode
from datetime import date, datetime
from typing import (
import numpy as np
import param
from bokeh.models.formatters import TickFormatter
from bokeh.models.widgets import (
from ..config import config
from ..layout import Column, Panel
from ..models import (
from ..util import param_reprs, try_datetime64_to_datetime
from .base import CompositeWidget, Widget
class LiteralInput(Widget):
    """
    The `LiteralInput` allows declaring Python literals using a text
    input widget.

    A *literal* is some specific primitive value of type `str`
    , `int`, `float`, `bool` etc or a `dict`, `list`, `tuple`, `set` etc of
    primitive values.

    Optionally the literal `type` may be declared.

    Reference: https://panel.holoviz.org/reference/widgets/LiteralInput.html

    :Example:

    >>> LiteralInput(name='Dictionary', value={'key': [1, 2, 3]}, type=dict)
    """
    description = param.String(default=None, doc='\n        An HTML string describing the function of this component.')
    placeholder = param.String(default='', doc='\n      Placeholder for empty input field.')
    serializer = param.ObjectSelector(default='ast', objects=['ast', 'json'], doc="\n       The serialization (and deserialization) method to use. 'ast'\n       uses ast.literal_eval and 'json' uses json.loads and json.dumps.\n    ")
    type = param.ClassSelector(default=None, class_=(type, tuple), is_instance=True)
    value = param.Parameter(default=None)
    width = param.Integer(default=300, allow_None=True, doc='\n      Width of this component. If sizing_mode is set to stretch\n      or scale mode this will merely be used as a suggestion.')
    _rename: ClassVar[Mapping[str, str | None]] = {'type': None, 'serializer': None}
    _source_transforms: ClassVar[Mapping[str, str | None]] = {'serializer': None, 'value': 'JSON.parse(value.replace(/\'/g, \'"\'))'}
    _target_transforms: ClassVar[Mapping[str, str | None]] = {'value': 'JSON.stringify(value).replace(/,/g, ",").replace(/:/g, ": ")'}
    _widget_type: ClassVar[Type[Model]] = _BkTextInput

    def __init__(self, **params):
        super().__init__(**params)
        self._state = ''
        self._validate(None)
        self._internal_callbacks.append(self.param.watch(self._validate, 'value'))

    def _validate(self, event):
        if self.type is None:
            return
        new = self.value
        if not isinstance(new, self.type) and new is not None:
            if event:
                self.value = event.old
            types = repr(self.type) if isinstance(self.type, tuple) else self.type.__name__
            raise ValueError('LiteralInput expected %s type but value %s is of type %s.' % (types, new, type(new).__name__))

    def _process_property_change(self, msg):
        msg = super()._process_property_change(msg)
        new_state = ''
        if 'value' in msg:
            value = msg.pop('value')
            try:
                if value == '':
                    value = ''
                elif self.serializer == 'json':
                    value = json.loads(value)
                else:
                    value = ast.literal_eval(value)
            except Exception:
                new_state = ' (invalid)'
                value = self.value
            else:
                if self.type and (not isinstance(value, self.type)):
                    vtypes = self.type if isinstance(self.type, tuple) else (self.type,)
                    typed_value = None
                    for vtype in vtypes:
                        try:
                            typed_value = vtype(value)
                        except Exception:
                            pass
                        else:
                            break
                    if typed_value is None and value == '':
                        value = None
                    elif typed_value is None and value is not None:
                        new_state = ' (wrong type)'
                        value = self.value
                    else:
                        value = typed_value
            msg['value'] = value
        msg['name'] = msg.get('title', self.name).replace(self._state, '') + new_state
        self._state = new_state
        self.param.trigger('name')
        return msg

    def _process_param_change(self, msg):
        msg = super()._process_param_change(msg)
        if 'value' in msg:
            value = msg['value']
            if isinstance(value, str):
                value = repr(value)
            elif self.serializer == 'json':
                value = json.dumps(value)
            else:
                value = '' if value is None else str(value)
            msg['value'] = value
        msg['title'] = self.name
        return msg