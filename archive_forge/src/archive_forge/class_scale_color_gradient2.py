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
class scale_color_gradient2(scale_continuous):
    """
    Create a 3 point diverging color gradient

    Parameters
    ----------
    low : str
        low color
    mid : str
        mid point color
    high : str
        high color
    midpoint : float, default=0
        Mid point of the input data range.
    {superclass_parameters}
    na_value : str, default="#7F7F7F"
        Color of missing values

    See Also
    --------
    plotnine.scale_color_gradient
    plotnine.scale_color_gradientn
    """
    _aesthetics = ['color']
    guide = 'colorbar'
    na_value = '#7F7F7F'

    def __init__(self, low='#832424', mid='#FFFFFF', high='#3A3A98', midpoint=0, **kwargs):
        from mizani.bounds import rescale_mid
        from mizani.palettes import gradient_n_pal

        def _rescale_mid(*args, **kwargs):
            return rescale_mid(*args, mid=midpoint, **kwargs)
        kwargs['rescaler'] = _rescale_mid
        self.palette = gradient_n_pal([low, mid, high])
        scale_continuous.__init__(self, **kwargs)