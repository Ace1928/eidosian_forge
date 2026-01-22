import os
import json
from warnings import warn
import ipywidgets as widgets
from ipywidgets import (Widget, DOMWidget, CallbackDispatcher,
from traitlets import (Int, Unicode, List, Enum, Dict, Bool, Float,
from traittypes import Array
from numpy import histogram
import numpy as np
from .scales import Scale, OrdinalScale, LinearScale
from .traits import (Date, array_serialization,
from ._version import __frontend_version__
from .colorschemes import CATEGORY10
@register_mark('bqplot.OHLC')
class OHLC(Mark):
    """Open/High/Low/Close marks.

    Attributes
    ----------
    icon: string (class-level attribute)
        font-awesome icon for that mark
    name: string (class-level attribute)
        user-friendly name of the mark
    marker: {'candle', 'bar'}
        marker type
    stroke: color (default: None)
        stroke color of the marker
    stroke_width: float (default: 1.0)
        stroke width of the marker
    colors: List of colors (default: ['limegreen', 'red'])
        fill colors for the markers (up/down)
    opacities: list of floats (default: [])
        Opacities for the markers of the OHLC mark. Defaults to 1 when
        the list is too short, or the element of the list is set to None.
    format: string (default: 'ohlc')
        description of y data being passed
        supports all permutations of the strings 'ohlc', 'oc', and 'hl'

    Data Attributes

    x: numpy.ndarray
        abscissas of the data points (1d array)
    y: numpy.ndarrays
        Open/High/Low/Close ordinates of the data points (2d array)

    Notes
    -----
    The fields which can be passed to the default tooltip are:
        x: the x value associated with the bar/candle
        open: open value for the bar/candle
        high: high value for the bar/candle
        low: low value for the bar/candle
        close: close value for the bar/candle
        index: index of the bar/candle being hovered on
    """
    icon = 'fa-birthday-cake'
    name = 'OHLC chart'
    x = Array([]).tag(sync=True, scaled=True, rtype='Number', atype='bqplot.Axis', **array_serialization).valid(array_squeeze, array_dimension_bounds(1, 1))
    y = Array([[]]).tag(sync=True, scaled=True, rtype='Number', atype='bqplot.Axis', **array_serialization).valid(array_dimension_bounds(1, 2))
    scales_metadata = Dict({'x': {'orientation': 'horizontal', 'dimension': 'x'}, 'y': {'orientation': 'vertical', 'dimension': 'y'}}).tag(sync=True)
    marker = Enum(['candle', 'bar'], default_value='candle').tag(sync=True, display_name='Marker')
    stroke = Color(None, allow_none=True).tag(sync=True, display_name='Stroke color')
    stroke_width = Float(1.0).tag(sync=True, display_name='Stroke Width')
    colors = List(trait=Color(default_value=None, allow_none=True), default_value=['green', 'red']).tag(sync=True, display_name='Colors')
    opacities = List(trait=Float(1.0, min=0, max=1, allow_none=True)).tag(sync=True, display_name='Opacities')
    format = Unicode('ohlc').tag(sync=True, display_name='Format')
    _view_name = Unicode('OHLC').tag(sync=True)
    _model_name = Unicode('OHLCModel').tag(sync=True)