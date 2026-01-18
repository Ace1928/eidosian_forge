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
def update_selection_expr(*_):
    new_selection_expr = inst.selection_expr
    current_selection_expr = inst._cross_filter_stream.selection_expr
    if repr(new_selection_expr) != repr(current_selection_expr):
        if inst.show_regions:
            inst.show_regions = False
        inst._selection_override.event(selection_expr=new_selection_expr)