from __future__ import annotations
import asyncio
import math
import os
import sys
import time
from math import pi
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource, FixedTicker, Tooltip
from bokeh.plotting import figure
from tqdm.asyncio import tqdm as _tqdm
from .._param import Align
from ..io.resources import CDN_DIST
from ..layout import Column, Panel, Row
from ..models import (
from ..pane.markup import Str
from ..reactive import SyncableData
from ..util import PARAM_NAME_PATTERN, escape, updating
from ..viewable import Viewable
from .base import Widget
class Dial(ValueIndicator):
    """
    A `Dial` represents a value in some range as a position on an
    annular dial. It is similar to a `Gauge` but more minimal
    visually.

    Reference: https://panel.holoviz.org/reference/indicators/Dial.html

    :Example:

    >>> Dial(name='Speed', value=79, format="{value} km/h", bounds=(0, 200), colors=[(0.4, 'green'), (1, 'red')])
    """
    annulus_width = param.Number(default=0.2, doc='\n      Width of the radial annulus as a fraction of the total.')
    background = param.Parameter(default=None, doc='\n        Background color of the component.')
    bounds = param.Range(default=(0, 100), doc='\n      The upper and lower bound of the dial.')
    colors = param.List(default=None, doc='\n      Color thresholds for the Dial, specified as a list of tuples\n      of the fractional threshold and the color to switch to.')
    default_color = param.String(default='lightblue', doc='\n      Color of the radial annulus if not color thresholds are supplied.')
    end_angle = param.Number(default=25, doc='\n      Angle at which the dial ends.')
    format = param.String(default='{value}%', doc='\n      Formatting string for the value indicator and lower/upper bounds.')
    height = param.Integer(default=250, bounds=(1, None))
    label_color = param.String(default='black', doc='\n      Color for all extraneous labels.')
    nan_format = param.String(default='-', doc='\n      How to format nan values.')
    needle_color = param.String(default='black', doc='\n      Color of the Dial needle.')
    needle_width = param.Number(default=0.1, doc='\n      Radial width of the needle.')
    start_angle = param.Number(default=-205, doc='\n      Angle at which the dial starts.')
    tick_size = param.String(default=None, doc='\n      Font size of the Dial min/max labels.')
    title_size = param.String(default=None, doc='\n      Font size of the Dial title.')
    unfilled_color = param.String(default='whitesmoke', doc='\n      Color of the unfilled region of the Dial.')
    value_size = param.String(default=None, doc='\n      Font size of the Dial value label.')
    value = param.Number(default=25, allow_None=True, doc='\n      Value to indicate on the dial a value within the declared bounds.')
    width = param.Integer(default=250, bounds=(1, None))
    _manual_params: ClassVar[List[str]] = ['value', 'start_angle', 'end_angle', 'bounds', 'annulus_width', 'format', 'background', 'needle_width', 'tick_size', 'title_size', 'value_size', 'colors', 'default_color', 'unfilled_color', 'height', 'width', 'nan_format', 'needle_color', 'label_color']
    _data_params: ClassVar[List[str]] = _manual_params
    _rename: ClassVar[Mapping[str, str | None]] = {'background': 'background_fill_color'}

    def __init__(self, **params):
        super().__init__(**params)
        self._update_value_bounds()

    @param.depends('bounds', watch=True)
    def _update_value_bounds(self):
        self.param.value.bounds = self.bounds

    def _get_data(self, properties):
        vmin, vmax = self.bounds
        value = self.value
        if value is None:
            value = float('nan')
        fraction = (value - vmin) / (vmax - vmin)
        start = np.radians(360 - self.start_angle) - pi % (2 * pi) + pi
        end = np.radians(360 - self.end_angle) - pi % (2 * pi) + pi
        distance = abs(end - start) % (pi * 2)
        if end > start:
            distance = pi * 2 - distance
        radial_fraction = distance * fraction
        angle = start if np.isnan(fraction) else start - radial_fraction
        inner_radius = 1 - self.annulus_width
        color = self.default_color
        for val, clr in (self.colors or [])[::-1]:
            if fraction <= val:
                color = clr
        annulus_data = {'starts': np.array([start, angle]), 'ends': np.array([angle, end]), 'color': [color, self.unfilled_color], 'radius': np.array([inner_radius, inner_radius])}
        x0s, y0s, x1s, y1s, clrs = ([], [], [], [], [])
        colors = self.colors or []
        for (val, _), (_, clr) in zip(colors[:-1], colors[1:]):
            tangle = start - distance * val
            if vmin + val * (vmax - vmin) <= value:
                continue
            x0, y0 = (np.cos(tangle), np.sin(tangle))
            x1, y1 = (x0 * inner_radius, y0 * inner_radius)
            x0s.append(x0)
            y0s.append(y0)
            x1s.append(x1)
            y1s.append(y1)
            clrs.append(clr)
        threshold_data = {'x0': x0s, 'y0': y0s, 'x1': x1s, 'y1': y1s, 'color': clrs}
        center_radius = 1 - self.annulus_width / 2.0
        x, y = (np.cos(angle) * center_radius, np.sin(angle) * center_radius)
        needle_start = pi + angle - self.needle_width / 2.0
        needle_end = pi + angle + self.needle_width / 2.0
        needle_data = {'x': np.array([x]), 'y': np.array([y]), 'start': np.array([needle_start]), 'end': np.array([needle_end]), 'radius': np.array([center_radius])}
        value = self.format.format(value=value).replace('nan', self.nan_format)
        min_value = self.format.format(value=vmin)
        max_value = self.format.format(value=vmax)
        tminx, tminy = (np.cos(start) * center_radius, np.sin(start) * center_radius)
        tmaxx, tmaxy = (np.cos(end) * center_radius, np.sin(end) * center_radius)
        tmin_angle, tmax_angle = (start + pi, end + pi % pi)
        scale = self.height / 400
        title_size = self.title_size if self.title_size else '%spt' % (scale * 32)
        value_size = self.value_size if self.value_size else '%spt' % (scale * 48)
        tick_size = self.tick_size if self.tick_size else '%spt' % (scale * 18)
        text_data = {'x': np.array([0, 0, tminx, tmaxx]), 'y': np.array([-0.2, -0.5, tminy, tmaxy]), 'text': [self.name, value, min_value, max_value], 'rot': np.array([0, 0, tmin_angle, tmax_angle]), 'size': [title_size, value_size, tick_size, tick_size], 'color': [self.label_color, color, self.label_color, self.label_color]}
        return (annulus_data, needle_data, threshold_data, text_data)

    def _get_model(self, doc, root=None, parent=None, comm=None):
        properties = self._get_properties(doc)
        model = figure(x_range=(-1, 1), y_range=(-1, 1), tools=[], outline_line_color=None, toolbar_location=None, width=self.width, height=self.height, **properties)
        model.xaxis.visible = False
        model.yaxis.visible = False
        model.grid.visible = False
        annulus, needle, threshold, text = self._get_data(properties)
        annulus_source = ColumnDataSource(data=annulus, name='annulus_source')
        model.annular_wedge(x=0, y=0, inner_radius='radius', outer_radius=1, start_angle='starts', end_angle='ends', line_color='gray', color='color', direction='clock', source=annulus_source)
        needle_source = ColumnDataSource(data=needle, name='needle_source')
        model.wedge(x='x', y='y', radius='radius', start_angle='start', end_angle='end', fill_color=self.needle_color, line_color=self.needle_color, source=needle_source, name='needle_renderer')
        threshold_source = ColumnDataSource(data=threshold, name='threshold_source')
        model.segment(x0='x0', x1='x1', y0='y0', y1='y1', line_color='color', source=threshold_source, line_width=2)
        text_source = ColumnDataSource(data=text, name='label_source')
        model.text(x='x', y='y', text='text', font_size='size', text_align='center', text_color='color', source=text_source, text_baseline='top', angle='rot')
        self._design.apply_bokeh_theme_to_model(model)
        if root is None:
            root = model
        self._models[root.ref['id']] = (model, parent)
        return model

    def _manual_update(self, events, model, doc, root, parent, comm):
        update_data = False
        for event in events:
            if event.name in ('width', 'height'):
                model.update(**{event.name: event.new})
            if event.name in self._data_params:
                update_data = True
            elif event.name == 'needle_color':
                needle_r = model.select(name='needle_renderer')
                needle_r.glyph.line_color = event.new
                needle_r.glyph.fill_color = event.new
        if not update_data:
            return
        properties = self._get_properties(doc)
        annulus, needle, threshold, labels = self._get_data(properties)
        model.select(name='annulus_source').data.update(annulus)
        model.select(name='needle_source').data.update(needle)
        model.select(name='threshold_source').data.update(threshold)
        model.select(name='label_source').data.update(labels)