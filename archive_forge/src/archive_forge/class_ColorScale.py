from ipywidgets import Widget, Color
from traitlets import Unicode, List, Enum, Float, Bool, Type, Tuple
import numpy as np
from .traits import Date
from ._version import __frontend_version__
@register_scale('bqplot.ColorScale')
class ColorScale(Scale):
    """A color scale.

    A mapping from numbers to colors. The relation is affine by part.

    Attributes
    ----------
    scale_type: {'linear'}
        scale type
    colors: list of colors (default: [])
        list of colors
    min: float or None (default: None)
        if not None, min is the minimal value of the domain
    max: float or None (default: None)
        if not None, max is the maximal value of the domain
    mid: float or None (default: None)
        if not None, mid is the value corresponding to the mid color.
    scheme: string (default: 'RdYlGn')
        Colorbrewer color scheme of the color scale.
    extrapolation: {'constant', 'linear'} (default: 'constant')
        How to extrapolate values outside the [min, max] domain.
    rtype: string (class-level attribute)
        The range type of a color scale is 'Color'. This should not be modified.
    dtype: type (class-level attribute)
        the associated data type / domain type
    """
    rtype = 'Color'
    dtype = np.number
    scale_type = Enum(['linear'], default_value='linear').tag(sync=True)
    colors = List(trait=Color(default_value=None, allow_none=True)).tag(sync=True)
    min = Float(None, allow_none=True).tag(sync=True)
    max = Float(None, allow_none=True).tag(sync=True)
    mid = Float(None, allow_none=True).tag(sync=True)
    scheme = Unicode('RdYlGn').tag(sync=True)
    extrapolation = Enum(['constant', 'linear'], default_value='constant').tag(sync=True)
    _view_name = Unicode('ColorScale').tag(sync=True)
    _model_name = Unicode('ColorScaleModel').tag(sync=True)