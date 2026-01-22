from __future__ import annotations
import datetime as dt
from typing import (
import numpy as np
import param
from bokeh.models import CustomJS
from bokeh.models.formatters import TickFormatter
from bokeh.models.widgets import (
from bokeh.models.widgets.sliders import NumericalSlider as _BkNumericalSlider
from param.parameterized import resolve_value
from ..config import config
from ..io import state
from ..io.resources import CDN_DIST
from ..layout import Column, Panel, Row
from ..util import (
from ..viewable import Layoutable
from ..widgets import FloatInput, IntInput
from .base import CompositeWidget, Widget
from .input import StaticText
class _SliderBase(Widget):
    bar_color = param.Color(default='#e6e6e6', doc='')
    direction = param.ObjectSelector(default='ltr', objects=['ltr', 'rtl'], doc="\n        Whether the slider should go from left-to-right ('ltr') or\n        right-to-left ('rtl').")
    name = param.String(default=None, doc='\n        The name of the widget. Also used as the label of the widget. If not set,\n        the widget has no label.')
    orientation = param.ObjectSelector(default='horizontal', objects=['horizontal', 'vertical'], doc='\n        Whether the slider should be oriented horizontally or\n        vertically.')
    show_value = param.Boolean(default=True, doc='\n        Whether to show the widget value as a label or not.')
    tooltips = param.Boolean(default=True, doc='\n        Whether the slider handle should display tooltips.')
    _widget_type: ClassVar[Type[Model]] = _BkSlider
    __abstract = True

    def __init__(self, **params):
        if 'value' in params and 'value_throttled' in self.param:
            params['value_throttled'] = params['value']
        if 'orientation' == 'vertical':
            params['height'] = self.param.width.default
        super().__init__(**params)

    def __repr__(self, depth=0):
        return '{cls}({params})'.format(cls=type(self).__name__, params=', '.join(param_reprs(self, ['value_throttled'])))

    @property
    def _linked_properties(self) -> Tuple[str]:
        return super()._linked_properties + ('value_throttled',)

    def _process_property_change(self, msg):
        if config.throttled:
            if 'value' in msg:
                del msg['value']
            if 'value_throttled' in msg:
                msg['value'] = msg['value_throttled']
        return super()._process_property_change(msg)

    def _update_model(self, events: Dict[str, param.parameterized.Event], msg: Dict[str, Any], root: Model, model: Model, doc: Document, comm: Optional[Comm]) -> None:
        if 'value_throttled' in msg:
            del msg['value_throttled']
        return super()._update_model(events, msg, root, model, doc, comm)