from warnings import warn
import numpy as np
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineWarning
from .scale_continuous import scale_continuous
from .scale_datetime import scale_datetime
from .scale_discrete import scale_discrete
@document
class scale_alpha_ordinal(scale_discrete):
    """
    Ordinal Alpha Scale

    Parameters
    ----------
    range :
        Range ([Minimum, Maximum]) of output alpha values.
        Should be between 0 and 1.
    {superclass_parameters}
    """
    _aesthetics = ['alpha']

    def __init__(self, range: tuple[float, float]=(0.1, 1), **kwargs):

        def palette(value):
            return np.linspace(range[0], range[1], value)
        self.palette = palette
        scale_discrete.__init__(self, **kwargs)