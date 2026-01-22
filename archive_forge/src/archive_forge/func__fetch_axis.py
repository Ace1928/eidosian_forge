import sys
from collections import OrderedDict
from IPython.display import display
from ipywidgets import VBox
from ipywidgets import Image as ipyImage
from numpy import arange, issubdtype, array, column_stack, shape
from .figure import Figure
from .scales import Scale, LinearScale, Mercator
from .axes import Axis
from .marks import (Lines, Scatter, ScatterGL, Hist, Bars, OHLC, Pie, Map, Image,
from .toolbar import Toolbar
from .interacts import (BrushIntervalSelector, FastIntervalSelector,
from traitlets.utils.sentinel import Sentinel
import functools
def _fetch_axis(fig, dimension, scale):
    axis_registry = getattr(fig, 'axis_registry', {})
    dimension_data = axis_registry.get(dimension, [])
    dimension_scales = [dim['scale'] for dim in dimension_data]
    dimension_axes = [dim['axis'] for dim in dimension_data]
    try:
        return dimension_axes[dimension_scales.index(scale)]
    except (ValueError, IndexError):
        return None