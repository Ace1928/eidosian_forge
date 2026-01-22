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
class DiscreteSlider(CompositeWidget, _SliderBase):
    """
    The DiscreteSlider widget allows selecting a value from a discrete
    list or dictionary of values using a slider.

    Reference: https://panel.holoviz.org/reference/widgets/DiscreteSlider.html

    :Example:

    >>> DiscreteSlider(
    ...     value=0,
    ...     options=list([0, 1, 2, 4, 8, 16, 32, 64]),
    ...     name="A discrete value",
    ... )
    """
    value = param.Parameter(doc='\n        The selected value of the slider. Updated when the handle is\n        dragged. Must be one of the options.')
    value_throttled = param.Parameter(constant=True, doc='\n        The value of the slider. Updated when the handle is released.')
    options = param.ClassSelector(default=[], class_=(dict, list), doc='\n        A list or dictionary of valid options.')
    formatter = param.String(default='%.3g', doc='\n        A custom format string. Separate from format parameter since\n        formatting is applied in Python, not via the bokeh TickFormatter.')
    _rename: ClassVar[Mapping[str, str | None]] = {'formatter': None}
    _source_transforms: ClassVar[Mapping[str, str | None]] = {'value': None, 'value_throttled': None, 'options': None}
    _supports_embed: ClassVar[bool] = True
    _style_params: ClassVar[List[str]] = [p for p in list(Layoutable.param) if p != 'name'] + ['orientation']
    _slider_style_params: ClassVar[List[str]] = ['bar_color', 'direction', 'disabled', 'orientation']
    _text_link = '\n    var labels = {labels}\n    target.text = labels[source.value]\n    '

    def __init__(self, **params):
        self._syncing = False
        super().__init__(**params)
        if 'formatter' not in params and all((isinstance(v, (int, np.int_)) for v in self.values)):
            self.formatter = '%d'
        if self.value is None and None not in self.values and self.options:
            self.value = self.values[0]
        elif self.value not in self.values and (not (self.value is None or self.options)):
            raise ValueError('Value %s not a valid option, ensure that the supplied value is one of the declared options.' % self.value)
        self._text = StaticText(margin=(5, 0, 0, 5), styles={'white-space': 'nowrap'})
        self._slider = None
        self._composite = Column(self._text, self._slider)
        self._update_options()
        self.param.watch(self._update_options, ['options', 'formatter', 'name'])
        self.param.watch(self._update_value, 'value')
        self.param.watch(self._update_value, 'value_throttled')
        self.param.watch(self._update_style, self._style_params)

    def _update_options(self, *events):
        values, labels = (self.values, self.labels)
        if not self.options and self.value is None:
            value = 0
            label = (f'{self.name}: ' if self.name else '') + '<b>-</b>'
        elif self.value not in values:
            value = 0
            self.value = values[0]
            label = labels[value]
        else:
            value = values.index(self.value)
            label = labels[value]
        disabled = True if len(values) in (0, 1) else self.disabled
        end = 1 if disabled else len(self.options) - 1
        self._slider = IntSlider(start=0, end=end, value=value, tooltips=False, show_value=False, margin=(0, 5, 5, 5), _supports_embed=False, disabled=disabled, **{p: getattr(self, p) for p in self._slider_style_params if p != 'disabled'})
        self._update_style()
        js_code = self._text_link.format(labels='[' + ', '.join([repr(l) for l in labels]) + ']')
        self._jslink = self._slider.jslink(self._text, code={'value': js_code})
        self._slider.param.watch(self._sync_value, 'value')
        self._slider.param.watch(self._sync_value, 'value_throttled')
        self.param.watch(self._update_slider_params, self._slider_style_params)
        self._text.value = label
        self._composite[1] = self._slider

    def _update_value(self, event):
        """
        This will update the IntSlider (behind the scene)
        based on changes to the DiscreteSlider (front).

        _syncing options is to avoid infinite loop.

        event.name is either value or value_throttled.
        """
        values = self.values
        if getattr(self, event.name) not in values:
            with param.edit_constant(self):
                setattr(self, event.name, values[0])
            return
        index = self.values.index(getattr(self, event.name))
        if event.name == 'value':
            self._text.value = self.labels[index]
        if self._syncing:
            return
        try:
            self._syncing = True
            with param.edit_constant(self._slider):
                setattr(self._slider, event.name, index)
        finally:
            self._syncing = False

    def _update_style(self, *events):
        style = {p: getattr(self, p) for p in self._style_params}
        margin = style.pop('margin')
        if isinstance(margin, tuple):
            if len(margin) == 2:
                t = b = margin[0]
                r = l = margin[1]
            else:
                t, r, b, l = margin
        else:
            t = r = b = l = margin
        text_margin = (t, 0, 0, l)
        slider_margin = (0, r, b, l)
        text_style = {k: v for k, v in style.items() if k not in ('style', 'orientation')}
        text_style['visible'] = self.show_value and text_style['visible']
        self._text.param.update(margin=text_margin, **text_style)
        self._slider.param.update(margin=slider_margin, **style)
        if self.width:
            style['width'] = self.width + l + r
        col_style = {k: v for k, v in style.items() if k != 'orientation'}
        self._composite.param.update(**col_style)

    def _update_slider_params(self, *events):
        style = {e.name: e.new for e in events}
        disabled = style.get('disabled', None)
        if disabled is False:
            if len(self.values) in (0, 1):
                self.param.warning('A DiscreteSlider can only be disabled if it has more than 1 option.')
                end = 1
                del style['disabled']
            else:
                end = len(self.options) - 1
            style['end'] = end
        self._slider.param.update(**style)

    def _sync_value(self, event):
        """
        This will update the DiscreteSlider (front)
        based on changes to the IntSlider (behind the scene).

        _syncing options is to avoid infinite loop.

        event.name is either value or value_throttled.
        """
        if self._syncing:
            return
        try:
            self._syncing = True
            with param.edit_constant(self):
                setattr(self, event.name, self.values[event.new])
        finally:
            self._syncing = False

    def _get_embed_state(self, root, values=None, max_opts=3):
        model = self._composite[1]._models[root.ref['id']][0]
        if values is None:
            values = self.values
        elif any((v not in self.values for v in values)):
            raise ValueError("Supplieed embed states were not found in the %s widgets' values list." % type(self).__name__)
        return (self, model, values, lambda x: x.value, 'value', 'cb_obj.value')

    @property
    def labels(self):
        """The list of labels to display"""
        title = self.name + ': ' if self.name else ''
        if isinstance(self.options, dict):
            return [title + '<b>%s</b>' % o for o in self.options]
        else:
            return [title + '<b>%s</b>' % (o if isinstance(o, str) else self.formatter % o) for o in self.options]

    @property
    def values(self):
        """The list of option values"""
        return list(self.options.values()) if isinstance(self.options, dict) else self.options