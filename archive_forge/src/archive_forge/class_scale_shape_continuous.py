from warnings import warn
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from .scale_continuous import scale_continuous
from .scale_discrete import scale_discrete
class scale_shape_continuous(scale_continuous):
    """
    Continuous scale for shapes

    This is not a valid type of scale.
    """

    def __init__(self):
        raise PlotnineError('A continuous variable can not be mapped to shape')