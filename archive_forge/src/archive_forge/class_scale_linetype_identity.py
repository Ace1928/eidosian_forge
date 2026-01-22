from __future__ import annotations
import typing
from .._utils.registry import alias
from ..doctools import document
from .scale_continuous import scale_continuous
from .scale_discrete import scale_discrete
@document
class scale_linetype_identity(MapTrainMixin, scale_discrete):
    """
    No linetype scaling

    Parameters
    ----------
    {superclass_parameters}
    guide : Optional[Literal["legend"]], default=None
        Whether to include a legend.
    """
    _aesthetics = ['linetype']