from traitlets import (Bool, Int, Float, Unicode, Dict,
from traittypes import Array
from ipywidgets import Widget, Color, widget_serialization, register
from .scales import Scale
from .traits import Date, array_serialization, _array_equal
from .marks import Lines
from ._version import __frontend_version__
import numpy as np
class TwoDSelector(Selector):
    """Two-dimensional selector interaction.

    Base class for all selectors which select data in both the x and y
    dimensions. The attributes 'x_scale' and 'y_scale' should be provided.

    Attributes
    ----------
    x_scale: An instance of Scale
        This is the scale which is used for inversion from the pixels to data
        coordinates in the x-direction. This scale is used for setting the
        selected attribute for the selector along with ``y_scale``.
    y_scale: An instance of Scale
        This is the scale which is used for inversion from the pixels to data
        coordinates in the y-direction. This scale is used for setting the
        selected attribute for the selector along with ``x_scale``.
    """
    x_scale = Instance(Scale, allow_none=True, default_value=None).tag(sync=True, dimension='x', **widget_serialization)
    y_scale = Instance(Scale, allow_none=True, default_value=None).tag(sync=True, dimension='y', **widget_serialization)
    _model_name = Unicode('TwoDSelectorModel').tag(sync=True)