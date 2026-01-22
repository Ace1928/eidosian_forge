from warnings import warn
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from .scale_continuous import scale_continuous
from .scale_discrete import scale_discrete
@document
class scale_shape_ordinal(scale_shape):
    """
    Scale for shapes

    Parameters
    ----------
    {superclass_parameters}
    """
    _aesthetics = ['shape']

    def __init__(self, **kwargs):
        warn('Using shapes for an ordinal variable is not advised.', PlotnineWarning)
        super().__init__(**kwargs)