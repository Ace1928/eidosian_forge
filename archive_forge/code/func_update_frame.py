import re
import uuid
import numpy as np
import param
from ... import Tiles
from ...core import util
from ...core.element import Element
from ...core.spaces import DynamicMap
from ...streams import Stream
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import dim_range_key
from .plot import PlotlyPlot
from .util import (
def update_frame(self, key, ranges=None, element=None, is_geo=False):
    reused = isinstance(self.hmap, DynamicMap) and self.overlaid
    self.prev_frame = self.current_frame
    if not reused and element is None:
        element = self._get_frame(key)
    elif element is not None:
        self.current_frame = element
        self.current_key = key
    items = [] if element is None else list(element.data.items())
    for _, el in items:
        if isinstance(el, Tiles):
            is_geo = True
    for k, subplot in self.subplots.items():
        if not (isinstance(self.hmap, DynamicMap) and element is not None):
            continue
        idx, _, _ = self._match_subplot(k, subplot, items, element)
        if idx is not None:
            items.pop(idx)
    if isinstance(self.hmap, DynamicMap) and items:
        self._create_dynamic_subplots(key, items, ranges)
    self.generate_plot(key, ranges, element, is_geo=is_geo)