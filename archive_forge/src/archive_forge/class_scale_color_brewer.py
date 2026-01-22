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
class scale_color_brewer(scale_discrete):
    """
    Sequential, diverging and qualitative discrete color scales

    See `colorbrewer.org <http://colorbrewer2.org/>`_

    Parameters
    ----------
    type :
        Type of data. Sequential, diverging or qualitative
    palette : int | str, default=1
         If a string, will use that named palette.
         If a number, will index into the list of palettes
         of appropriate type.
    direction: 1 | -1, default=1
         Sets the order of colors in the scale. If 1, colors are
         as output by [](`~mizani.palettes.brewer_pal`). If -1,
         the order of colors is reversed.
    {superclass_parameters}
    na_value : str, default="#7F7F7F"
        Color of missing values.
    """
    _aesthetics = ['color']
    na_value = '#7F7F7F'

    def __init__(self, type: ColorScheme | ColorSchemeShort='seq', palette: int | str=1, direction: Literal[1, -1]=1, **kwargs):
        from mizani.palettes import brewer_pal
        self._palette = brewer_pal(type, palette, direction=direction)
        scale_discrete.__init__(self, **kwargs)