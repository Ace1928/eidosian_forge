from collections.abc import Sequence
import numpy as np
import xarray as xr
from numpy.linalg import LinAlgError
from scipy import special, stats
from . import _remove_indexes_to_reduce
from .linalg import cholesky, eigh
class XrContinuousRV(XrRV):
    """Wrapper for subclasses of :class:`~scipy.stats.rv_continuous`.

    Usage examples available at :ref:`stats_tutorial`

    See Also
    --------
    xarray_einstats.stats.XrDiscreteRV

    Examples
    --------
    Evaluate the ppf of a Student-T distribution from DataArrays that need
    broadcasting:

    .. jupyter-execute::

        from xarray_einstats import tutorial
        from xarray_einstats.stats import XrContinuousRV
        from scipy import stats
        ds = tutorial.generate_mcmc_like_dataset(3)
        dist = XrContinuousRV(stats.t, 3, ds["mu"], ds["sigma"])
        dist.ppf([.1, .5, .6])

    """