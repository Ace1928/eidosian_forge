from a list that contains the shared attribute. This is required for the
import abc
import operator
import sys
from functools import partial
from packaging.version import Version
from types import FunctionType, MethodType
import holoviews as hv
import pandas as pd
import panel as pn
import param
from panel.layout import Column, Row, VSpacer, HSpacer
from panel.util import get_method_owner, full_groupby
from panel.widgets.base import Widget
from .converter import HoloViewsConverter
from .util import (
def _layout_bk2(self, **kwargs):
    widget_box = self.widgets()
    panel = self.output()
    loc = self._loc
    if loc in ('left', 'right'):
        widgets = Column(VSpacer(), widget_box, VSpacer())
    elif loc in ('top', 'bottom'):
        widgets = Row(HSpacer(), widget_box, HSpacer())
    elif loc in ('top_left', 'bottom_left'):
        widgets = Row(widget_box, HSpacer())
    elif loc in ('top_right', 'bottom_right'):
        widgets = Row(HSpacer(), widget_box)
    elif loc in ('left_top', 'right_top'):
        widgets = Column(widget_box, VSpacer())
    elif loc in ('left_bottom', 'right_bottom'):
        widgets = Column(VSpacer(), widget_box)
    center = self._center
    if not widgets:
        if center:
            components = [HSpacer(), panel, HSpacer()]
        else:
            components = [panel]
    elif center:
        if loc.startswith('left'):
            components = [widgets, HSpacer(), panel, HSpacer()]
        elif loc.startswith('right'):
            components = [HSpacer(), panel, HSpacer(), widgets]
        elif loc.startswith('top'):
            components = [HSpacer(), Column(widgets, Row(HSpacer(), panel, HSpacer())), HSpacer()]
        elif loc.startswith('bottom'):
            components = [HSpacer(), Column(Row(HSpacer(), panel, HSpacer()), widgets), HSpacer()]
    elif loc.startswith('left'):
        components = [widgets, panel]
    elif loc.startswith('right'):
        components = [panel, widgets]
    elif loc.startswith('top'):
        components = [Column(widgets, panel)]
    elif loc.startswith('bottom'):
        components = [Column(panel, widgets)]
    return Row(*components, **kwargs)