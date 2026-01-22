import sys
from importlib.util import find_spec
import numpy as np
import pandas as pd
from ..core import Dataset, NdOverlay, util
from ..streams import Lasso, Selection1D, SelectionXY
from ..util.transform import dim
from .annotation import HSpan, VSpan
def _empty_region(self):
    invert_axes = self.opts.get('plot').kwargs.get('invert_axes', False)
    if invert_axes and (not self._inverted_expr) or (not invert_axes and self._inverted_expr):
        region_el = HSpan
    else:
        region_el = VSpan
    return NdOverlay({0: region_el()})