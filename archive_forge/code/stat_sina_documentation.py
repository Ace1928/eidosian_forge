import numpy as np
import pandas as pd
from .._utils import array_kind, jitter, resolution
from ..doctools import document
from ..exceptions import PlotnineError
from ..mapping.aes import has_groups
from .binning import breaks_from_bins, breaks_from_binwidth
from .stat import stat
from .stat_density import compute_density

    Compute Sina plot values

    {usage}

    Parameters
    ----------
    {common_parameters}
    binwidth : float, default=None
        The width of the bins. The default is to use bins that
        cover the range of the data. You should always override this
        value, exploring multiple widths to find the best to
        illustrate the stories in your data.
    bins : int, default=50
        Number of bins. Overridden by binwidth.
    method : Literal["density", "counts"], default="density"
        Choose the method to spread the samples within the same bin
        along the x-axis. Available methods: "density", "counts"
        (can be abbreviated, e.g. "d"). See Details.
    maxwidth : float, default=None
        Control the maximum width the points can spread into.
        Values should be in the range (0, 1).
    adjust : float, default=1
        Adjusts the bandwidth of the density kernel when
        `method="density"`. see [](`~plotnine.stats.stat_density`).
    bw : str | float, default="nrd0"
        The bandwidth to use, If a float is given, it is the bandwidth.
        The `str`{.py} choices are:
        `"nrd0", "normal_reference", "scott", "silverman"`{.py}

        `nrd0` is a port of `stats::bw.nrd0` in R; it is eqiuvalent
        to `silverman` when there is more than 1 value in a group.
    bin_limit : int, default=1
        If the samples within the same y-axis bin are more
        than `bin_limit`, the samples's X coordinates will be adjusted.
        This parameter is effective only when `method="counts"`{.py}
    random_state : int | ~numpy.random.RandomState, default=None
        Seed or Random number generator to use. If `None`, then
        numpy global generator [](`numpy.random`) is used.
    scale : Literal["area", "count", "width"], default="area"
        How to scale the sina groups.

        - `area` - Scale by the largest density/bin among the different sinas
        - `count` - areas are scaled proportionally to the number of points
        - `width` - Only scale according to the maxwidth parameter.

    See Also
    --------
    plotnine.geom_sina
    