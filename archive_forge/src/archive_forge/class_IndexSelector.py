from traitlets import (Bool, Int, Float, Unicode, Dict,
from traittypes import Array
from ipywidgets import Widget, Color, widget_serialization, register
from .scales import Scale
from .traits import Date, array_serialization, _array_equal
from .marks import Lines
from ._version import __frontend_version__
import numpy as np
@register_interaction('bqplot.IndexSelector')
class IndexSelector(OneDSelector):
    """Index selector interaction.

    This 1-D selector interaction uses the mouse x-coordinate to select the
    corresponding point in terms of the selector scale.

    Index Selector has two modes:
        1. default mode: The mouse controls the x-position of the selector.
        2. frozen mode: In this mode, the selector is frozen at a point and
                does not respond to mouse events.

        A single click switches between the two modes.

    Attributes
    ----------
    selected: numpy.ndarray
        A single element array containing the point corresponding the
        x-position of the mouse. This attribute is updated as you move the
        mouse along the x-direction on the figure.
    color: Color or None (default: None)
        Color of the line representing the index selector.
    line_width: nonnegative integer (default: 0)
        Width of the line represetning the index selector.
    """
    selected = Array(None, allow_none=True).tag(sync=True, **array_serialization)
    line_width = Int(2).tag(sync=True)
    color = Color(None, allow_none=True).tag(sync=True)
    _view_name = Unicode('IndexSelector').tag(sync=True)
    _model_name = Unicode('IndexSelectorModel').tag(sync=True)