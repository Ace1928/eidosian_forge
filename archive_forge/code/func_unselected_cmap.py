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
@property
def unselected_cmap(self):
    """
        The datashader colormap for unselected data
        """
    if self.unselected_color is None:
        return None
    return _color_to_cmap(self.unselected_color)