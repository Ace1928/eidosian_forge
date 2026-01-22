from collections import namedtuple
import numpy as np
import param
from param.parameterized import bothmethod
from .core.data import Dataset
from .core.element import Element, Layout
from .core.layout import AdjointLayout
from .core.options import CallbackError, Store
from .core.overlay import NdOverlay, Overlay
from .core.spaces import GridSpace
from .streams import (
from .util import DynamicMap
def _apply_style_callback(self, element, layer_number, colors, cmap, alpha, **kwargs):
    opts = {}
    if layer_number == 0:
        opts['colorbar'] = False
    else:
        alpha = 1
    if cmap is not None:
        opts['cmap'] = cmap
    color = colors[layer_number] if colors else None
    return self._build_element_layer(element, color, alpha, **opts)