from __future__ import annotations
import typing
from warnings import warn
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineWarning
from .scale_continuous import scale_continuous
from .scale_datetime import scale_datetime
from .scale_discrete import scale_discrete
@document
class scale_fill_gradient2(scale_color_gradient2):
    """
    Create a 3 point diverging color gradient

    Parameters
    ----------
    {superclass_parameters}
    """
    _aesthetics = ['fill']