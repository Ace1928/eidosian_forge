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
class scale_color_gradient(scale_continuous):
    """
    Create a 2 point color gradient

    Parameters
    ----------
    low : str
        low color
    high : str
        high color
    {superclass_parameters}
    na_value : str, default="#7F7F7F"
        Color of missing values.

    See Also
    --------
    plotnine.scale_color_gradient2
    plotnine.scale_color_gradientn
    """
    _aesthetics = ['color']
    guide = 'colorbar'
    na_value = '#7F7F7F'

    def __init__(self, low='#132B43', high='#56B1F7', **kwargs):
        """
        Create colormap that will be used by the palette
        """
        from mizani.palettes import gradient_n_pal
        self._palette = gradient_n_pal([low, high])
        scale_continuous.__init__(self, **kwargs)