from warnings import warn
import numpy as np
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineWarning
from .scale_continuous import scale_continuous
from .scale_datetime import scale_datetime
from .scale_discrete import scale_discrete
@document
class scale_alpha_discrete(scale_alpha_ordinal):
    """
    Discrete Alpha Scale

    Parameters
    ----------
    {superclass_parameters}
    """
    _aesthetics = ['alpha']

    def __init__(self, **kwargs):
        warn('Using alpha for a discrete variable is not advised.', PlotnineWarning)
        super().__init__(**kwargs)