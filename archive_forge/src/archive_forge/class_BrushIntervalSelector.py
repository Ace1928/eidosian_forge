from traitlets import (Bool, Int, Float, Unicode, Dict,
from traittypes import Array
from ipywidgets import Widget, Color, widget_serialization, register
from .scales import Scale
from .traits import Date, array_serialization, _array_equal
from .marks import Lines
from ._version import __frontend_version__
import numpy as np
@register_interaction('bqplot.BrushIntervalSelector')
class BrushIntervalSelector(OneDSelector):
    """Brush interval selector interaction.

    This 1-D selector interaction enables the user to select an interval using
    the brushing action of the mouse. A mouse-down marks the start of the
    interval. The drag after the mouse down in the x-direction selects the
    extent and a mouse-up signifies the end of the interval.

    Once an interval is drawn, the selector can be moved to a new interval by
    dragging the selector to the new interval.

    A double click at the same point without moving the mouse in the
    x-direction will result in the entire interval being selected.

    Attributes
    ----------
    selected: numpy.ndarray
        Two element array containing the start and end of the interval selected
        in terms of the scale of the selector.
        This attribute changes while the selection is being made with the
        ``BrushIntervalSelector``.
    brushing: bool
        Boolean attribute to indicate if the selector is being dragged.
        It is True when the selector is being moved and False when it is not.
        This attribute can be used to trigger computationally intensive code
        which should be run only on the interval selection being completed as
        opposed to code which should be run whenever selected is changing.
    orientation: {'horizontal', 'vertical'}
        The orientation of the interval, either vertical or horizontal
    color: Color or None (default: None)
        Color of the rectangle representing the brush selector.
    """
    brushing = Bool().tag(sync=True)
    selected = Array(None, allow_none=True).tag(sync=True, **array_serialization)
    orientation = Enum(['horizontal', 'vertical'], default_value='horizontal').tag(sync=True)
    color = Color(None, allow_none=True).tag(sync=True)
    _view_name = Unicode('BrushIntervalSelector').tag(sync=True)
    _model_name = Unicode('BrushIntervalSelectorModel').tag(sync=True)