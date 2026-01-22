from warnings import warn
import numpy as np
from mizani.bounds import rescale_max
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineWarning
from .scale_continuous import scale_continuous
from .scale_datetime import scale_datetime
from .scale_discrete import scale_discrete
@document
class scale_size_ordinal(scale_discrete):
    """
    Discrete area size scale

    Parameters
    ----------
    range :
        Minimum and maximum size of the plotting symbol.
        It must be of size 2.
    {superclass_parameters}
    """
    _aesthetics = ['size']

    def __init__(self, range: tuple[float, float]=(2, 6), **kwargs):

        def palette(value):
            area = np.linspace(range[0] ** 2, range[1] ** 2, value)
            return np.sqrt(area)
        self.palette = palette
        scale_discrete.__init__(self, **kwargs)