from __future__ import annotations
import typing
from ..doctools import document
from .geom_ribbon import geom_ribbon

    Area plot

    An area plot is a special case of geom_ribbon,
    where the minimum of the range is fixed to 0,
    and the position adjustment defaults to 'stack'.

    {usage}

    Parameters
    ----------
    {common_parameters}

    See Also
    --------
    plotnine.geom_ribbon
    