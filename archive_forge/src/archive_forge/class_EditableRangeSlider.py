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
class EditableRangeSlider(CompositeWidget, _SliderBase):
    """
    The EditableRangeSlider widget allows selecting a floating-point
    range using a slider with two handles and for more precise control
    also offers a set of number input boxes.

    Reference: https://panel.holoviz.org/reference/widgets/EditableRangeSlider.html

    :Example:

    >>> EditableRangeSlider(
    ...      value=(1.0, 1.5), start=0.0, end=2.0, step=0.25, name="A tuple of floats"
    ... )
    """
    value = param.Range(default=(0, 1), allow_None=False, doc='\n        Current range value. Updated when a handle is dragged.')
    value_throttled = param.Range(default=None, constant=True, doc='\n        The value of the slider. Updated when the handle is released.')
    start = param.Number(default=0.0, doc='Lower bound of the range.')
    end = param.Number(default=1.0, doc='Upper bound of the range.')
    fixed_start = param.Number(default=None, doc='\n        A fixed lower bound for the slider and input.')
    fixed_end = param.Number(default=None, doc='\n        A fixed upper bound for the slider and input.')
    step = param.Number(default=0.1, doc='Slider and number input step.')
    editable = param.Tuple(default=(True, True), doc='\n        Whether the lower and upper values are editable.')
    format = param.ClassSelector(default='0.0[0000]', class_=(str, TickFormatter), doc='\n        Allows defining a custom format string or bokeh TickFormatter.')
    show_value = param.Boolean(default=False, readonly=True, precedence=-1, doc='\n        Whether to show the widget value.')
    _composite_type: ClassVar[Type[Panel]] = Column

    def __init__(self, **params):
        if 'width' not in params and 'sizing_mode' not in params:
            params['width'] = 300
        self._validate_init_bounds(params)
        super().__init__(**params)
        self._label = StaticText(margin=0, align='end')
        self._slider = RangeSlider(margin=(0, 0, 5, 0), show_value=False)
        self._slider.param.watch(self._sync_value, 'value')
        self._slider.param.watch(self._sync_value, 'value_throttled')
        self._start_edit = FloatInput(css_classes=['slider-edit'], stylesheets=[f'{CDN_DIST}css/editable_slider.css'], min_width=50, margin=0, format=self.format)
        self._end_edit = FloatInput(css_classes=['slider-edit'], stylesheets=[f'{CDN_DIST}css/editable_slider.css'], min_width=50, margin=(0, 0, 0, 10), format=self.format)
        self._start_edit.param.watch(self._sync_start_value, 'value')
        self._start_edit.param.watch(self._sync_start_value, 'value_throttled')
        self._end_edit.param.watch(self._sync_end_value, 'value')
        self._end_edit.param.watch(self._sync_end_value, 'value_throttled')
        sep = StaticText(value='...', margin=(0, 5, 0, 5), align='end')
        edit = Row(self._label, self._start_edit, sep, self._end_edit, sizing_mode='stretch_width', margin=0)
        self._composite.extend([edit, self._slider])
        self._start_edit.jscallback(args={'slider': self._slider, 'end': self._end_edit}, value='\n        // start value always smaller than the end value\n        if (cb_obj.value >= end.value) {\n          cb_obj.value = end.value\n          return\n        }\n        if (cb_obj.value < slider.start) {\n          slider.start = cb_obj.value\n        } else if (cb_obj.value > slider.end) {\n          slider.end = cb_obj.value\n        }\n        ')
        self._end_edit.jscallback(args={'slider': self._slider, 'start': self._start_edit}, value='\n        // end value always larger than the start value\n        if (cb_obj.value <= start.value) {\n          cb_obj.value = start.value\n          return\n        }\n        if (cb_obj.value < slider.start) {\n          slider.start = cb_obj.value\n        } else if (cb_obj.value > slider.end) {\n          slider.end = cb_obj.value\n        }\n        ')
        self._update_editable()
        self._update_disabled()
        self._update_layout()
        self._update_name()
        self._update_slider()
        self._update_value()
        self._update_bounds()

    def _validate_init_bounds(self, params):
        """
        This updates the default value, start and end
        if outside the fixed_start and fixed_end
        """
        start, end = (None, None)
        if 'start' not in params:
            if 'fixed_start' in params:
                start = params['fixed_start']
            elif 'end' in params:
                start = params.get('end') - params.get('step', 1)
            elif 'fixed_end' in params:
                start = params.get('fixed_end') - params.get('step', 1)
        if 'end' not in params:
            if 'fixed_end' in params:
                end = params['fixed_end']
            elif 'start' in params:
                end = params['start'] + params.get('step', 1)
            elif 'fixed_start' in params:
                end = params['fixed_start'] + params.get('step', 1)
        if start is not None:
            params['start'] = start
        if end is not None:
            params['end'] = end
        if 'value' not in params and 'start' in params:
            start = params['start']
            end = params.get('end', start + params.get('step', 1))
            params['value'] = (start, end)
        if 'value' not in params and 'end' in params:
            end = params['end']
            start = params.get('start', end - params.get('step', 1))
            params['value'] = (start, end)

    @param.depends('disabled', watch=True)
    def _update_disabled(self):
        self._slider.disabled = self.disabled

    @param.depends('disabled', 'editable', watch=True)
    def _update_editable(self):
        self._start_edit.disabled = not self.editable[0] or self.disabled
        self._end_edit.disabled = not self.editable[1] or self.disabled

    @param.depends('name', watch=True)
    def _update_name(self):
        if self.name:
            label = f'{self.name}:'
            margin = (0, 10, 0, 0)
        else:
            label = ''
            margin = (0, 0, 0, 0)
        self._label.param.update(margin=margin, value=label)

    @param.depends('width', 'height', 'sizing_mode', watch=True)
    def _update_layout(self):
        self._start_edit.sizing_mode = self.sizing_mode
        self._end_edit.sizing_mode = self.sizing_mode
        if self.sizing_mode not in ('stretch_width', 'stretch_both'):
            w = (self.width or 300) // 4
            self._start_edit.width = w
            self._end_edit.width = w

    @param.depends('start', 'end', 'step', 'bar_color', 'direction', 'show_value', 'tooltips', 'name', 'format', watch=True)
    def _update_slider(self):
        self._slider.param.update(format=self.format, start=self.start, end=self.end, step=self.step, bar_color=self.bar_color, direction=self.direction, show_value=self.show_value, tooltips=self.tooltips)
        self._start_edit.step = self.step
        self._end_edit.step = self.step

    @param.depends('value', watch=True)
    def _update_value(self):
        self._slider.value = self.value
        self._start_edit.value = self.value[0]
        self._end_edit.value = self.value[1]

    def _sync_value(self, event):
        with param.edit_constant(self):
            self.param.update(**{event.name: event.new})

    def _sync_start_value(self, event):
        if event.name == 'value':
            end = self.value[1] if self.value else self.end
        else:
            end = self.value_throttled[1] if self.value_throttled else self.end
        with param.edit_constant(self):
            self.param.update(**{event.name: (event.new, end)})

    def _sync_end_value(self, event):
        if event.name == 'value':
            start = self.value[0] if self.value else self.start
        else:
            start = self.value_throttled[0] if self.value_throttled else self.start
        with param.edit_constant(self):
            self.param.update(**{event.name: (start, event.new)})

    @param.depends('start', 'end', 'fixed_start', 'fixed_end', watch=True)
    def _update_bounds(self):
        self.param.value.softbounds = (self.start, self.end)
        self.param.value_throttled.softbounds = (self.start, self.end)
        self.param.value.bounds = (self.fixed_start, self.fixed_end)
        self.param.value_throttled.bounds = (self.fixed_start, self.fixed_end)
        if self.fixed_start is not None:
            self._slider.start = max(self.fixed_start, self.start)
        if self.fixed_end is not None:
            self._slider.end = min(self.fixed_end, self.end)
        self._start_edit.start = self.fixed_start
        self._start_edit.end = self.fixed_end
        self._end_edit.start = self.fixed_start
        self._end_edit.end = self.fixed_end