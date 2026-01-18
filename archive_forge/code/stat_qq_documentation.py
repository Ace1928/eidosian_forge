import numpy as np
import pandas as pd
from ..doctools import document
from ..exceptions import PlotnineError
from ..mapping.evaluation import after_stat
from .stat import stat

    Calculation for quantile-quantile plot

    {usage}

    Parameters
    ----------
    {common_parameters}
    distribution : str, default="norm"
        Distribution or distribution function name. The default is
        *norm* for a normal probability plot. Objects that look enough
        like a stats.distributions instance (i.e. they have a ppf
        method) are also accepted. See [scipy stats](`scipy.stats`)
        for available distributions.
    dparams : dict, default=None
        Distribution-specific shape parameters (shape parameters plus
        location and scale).
    quantiles : array_like, default=None
        Probability points at which to calculate the theoretical
        quantile values. If provided, must be the same number as
        as the sample data points. The default is to use calculated
        theoretical points, use to `alpha_beta` control how
        these points are generated.
    alpha_beta : tuple, default=(3/8, 3/8)
        Parameter values to use when calculating the quantiles.

    See Also
    --------
    scipy.stats.mstats.plotting_positions : Uses `alpha_beta`
        to calculate the quantiles.
    