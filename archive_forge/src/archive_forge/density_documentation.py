from __future__ import annotations
import typing
import numpy as np
from .._utils import array_kind

    Return var_type (for KDEMultivariate) of the column

    Parameters
    ----------
    col :
        A dataframe column.

    Returns
    -------
    out :
        Character that denotes the type of column.
        `c` for continuous, `o` for ordered categorical and
        `u` for unordered categorical or if not sure.

    See Also
    --------
    statsmodels.nonparametric.kernel_density.KDEMultivariate : For the origin
        of the character codes.
    