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
def _apply_properties(widget, properties={}):
    """Applies the specified properties to the widget.

    `properties` is a dictionary with key value pairs corresponding
    to the properties to be applied to the widget.
    """
    with widget.hold_sync():
        for key, value in properties.items():
            setattr(widget, key, value)