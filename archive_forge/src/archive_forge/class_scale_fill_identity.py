from __future__ import annotations
import typing
from .._utils.registry import alias
from ..doctools import document
from .scale_continuous import scale_continuous
from .scale_discrete import scale_discrete
@document
class scale_fill_identity(scale_color_identity):
    """
    No color scaling

    Parameters
    ----------
    {superclass_parameters}
    """
    _aesthetics = ['fill']