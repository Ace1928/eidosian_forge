import sys
import weakref
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import groupby
from numbers import Number
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
from .core import util
from .core.ndmapping import UniformNdMapping
class PointDraw(CDSStream):
    """
    Attaches a PointDrawTool and syncs the datasource.

    add: boolean
        Whether to allow adding new Points

    drag: boolean
        Whether to enable dragging of Points

    empty_value: int/float/string/None
        The value to insert on non-position columns when adding a new polygon

    num_objects: int
        The number of polygons that can be drawn before overwriting
        the oldest polygon.

    styles: dict
        A dictionary specifying lists of styles to cycle over whenever
        a new Point glyph is drawn.

    tooltip: str
        An optional tooltip to override the default
    """
    data = param.Dict(constant=True, doc='\n        Data synced from Bokeh ColumnDataSource supplied as a\n        dictionary of columns, where each column is a list of values\n        (for point-like data) or list of lists of values (for\n        path-like data).')

    def __init__(self, empty_value=None, add=True, drag=True, num_objects=0, styles=None, tooltip=None, **params):
        if styles is None:
            styles = {}
        self.add = add
        self.drag = drag
        self.empty_value = empty_value
        self.num_objects = num_objects
        self.styles = styles
        self.tooltip = tooltip
        self.styles = styles
        super().__init__(**params)

    @property
    def element(self):
        source = self.source
        if isinstance(source, UniformNdMapping):
            source = source.last
        if not self.data:
            return source.clone([], id=None)
        return source.clone(self.data, id=None)

    @property
    def dynamic(self):
        from .core.spaces import DynamicMap
        return DynamicMap(lambda *args, **kwargs: self.element, streams=[self])