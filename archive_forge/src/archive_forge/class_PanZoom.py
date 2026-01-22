from traitlets import (Bool, Int, Float, Unicode, Dict,
from traittypes import Array
from ipywidgets import Widget, Color, widget_serialization, register
from .scales import Scale
from .traits import Date, array_serialization, _array_equal
from .marks import Lines
from ._version import __frontend_version__
import numpy as np
@register_interaction('bqplot.PanZoom')
@register
class PanZoom(Interaction):
    """An interaction to pan and zoom wrt scales.

    Attributes
    ----------
    allow_pan: bool (default: True)
        Toggle the ability to pan.
    allow_zoom: bool (default: True)
        Toggle the ability to zoom.
    scales: Dictionary of lists of Scales (default: {})
        Dictionary with keys such as 'x' and 'y' and values being the scales in
        the corresponding direction (dimensions) which should be panned or
        zoomed.
    """
    allow_pan = Bool(True).tag(sync=True)
    allow_zoom = Bool(True).tag(sync=True)
    scales = Dict(value_trait=List(trait=Instance(Scale))).tag(sync=True, **widget_serialization)
    _view_name = Unicode('PanZoom').tag(sync=True)
    _model_name = Unicode('PanZoomModel').tag(sync=True)