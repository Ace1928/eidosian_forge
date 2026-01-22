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
def _update_mode(self, event):
    if event.new == 'replace':
        self.selection_mode = 'overwrite'
    elif event.new == 'append':
        self.selection_mode = 'union'
    elif event.new == 'intersect':
        self.selection_mode = 'intersect'
    elif event.new == 'subtract':
        self.selection_mode = 'inverse'