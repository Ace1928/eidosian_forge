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
class scale_color_grey(scale_discrete):
    """
    Sequential grey color scale.

    Parameters
    ----------
    start : float, default=0.2
        grey value at low end of palette.
    end : float, default=0.8
        grey value at high end of palette
    {superclass_parameters}
    """
    _aesthetics = ['color']

    def __init__(self, start=0.2, end=0.8, **kwargs):
        from mizani.palettes import grey_pal
        self._palette = grey_pal(start, end)
        scale_discrete.__init__(self, **kwargs)