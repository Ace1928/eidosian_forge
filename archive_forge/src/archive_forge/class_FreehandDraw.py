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
class FreehandDraw(CDSStream):
    """
    Attaches a FreehandDrawTool and syncs the datasource.

    empty_value: int/float/string/None
        The value to insert on non-position columns when adding a new polygon

    num_objects: int
        The number of polygons that can be drawn before overwriting
        the oldest polygon.

    styles: dict
        A dictionary specifying lists of styles to cycle over whenever
        a new freehand glyph is drawn.

    tooltip: str
        An optional tooltip to override the default
    """
    data = param.Dict(constant=True, doc='\n        Data synced from Bokeh ColumnDataSource supplied as a\n        dictionary of columns, where each column is a list of values\n        (for point-like data) or list of lists of values (for\n        path-like data).')

    def __init__(self, empty_value=None, num_objects=0, styles=None, tooltip=None, **params):
        if styles is None:
            styles = {}
        self.empty_value = empty_value
        self.num_objects = num_objects
        self.styles = styles
        self.tooltip = tooltip
        super().__init__(**params)

    @property
    def element(self):
        source = self.source
        if isinstance(source, UniformNdMapping):
            source = source.last
        data = self.data
        if not data:
            return source.clone([], id=None)
        cols = list(self.data)
        x, y = source.kdims
        lookup = {'xs': x.name, 'ys': y.name}
        data = [{lookup.get(c, c): data[c][i] for c in self.data} for i in range(len(data[cols[0]]))]
        return source.clone(data, id=None)

    @property
    def dynamic(self):
        from .core.spaces import DynamicMap
        return DynamicMap(lambda *args, **kwargs: self.element, streams=[self])