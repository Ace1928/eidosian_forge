from warnings import warn
import numpy as np
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineWarning
from .scale_continuous import scale_continuous
from .scale_discrete import scale_discrete
@document
class scale_stroke_discrete(scale_stroke_ordinal):
    """
    Discrete Stroke Scale

    Parameters
    ----------
    {superclass_parameters}
    """
    _aesthetics = ['stroke']

    def __init__(self, **kwargs):
        warn('Using stroke for a ordinal variable is not advised.', PlotnineWarning)
        super().__init__(**kwargs)