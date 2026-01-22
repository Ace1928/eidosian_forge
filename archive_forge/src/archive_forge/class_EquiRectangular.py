from ipywidgets import Widget, Color
from traitlets import Unicode, List, Enum, Float, Bool, Type, Tuple
import numpy as np
from .traits import Date
from ._version import __frontend_version__
@register_scale('bqplot.EquiRectangular')
class EquiRectangular(GeoScale):
    """An elementary projection that uses the identity function.

    The projection is neither equal-area nor conformal.

    Attributes
    ----------
    scale_factor: float (default: 145)
       Specifies the scale value for the projection
    center: tuple (default: (0, 60))
        Specifies the longitude and latitude where the map is centered.
    """
    scale_factor = Float(145.0).tag(sync=True)
    center = Tuple((0, 60)).tag(sync=True)
    rtype = '(Number, Number)'
    dtype = np.number
    _view_name = Unicode('EquiRectangular').tag(sync=True)
    _model_name = Unicode('EquiRectangularModel').tag(sync=True)