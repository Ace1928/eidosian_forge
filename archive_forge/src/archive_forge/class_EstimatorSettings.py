import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from . import kernels
class EstimatorSettings:
    """
    Object to specify settings for density estimation or regression.

    `EstimatorSettings` has several properties related to how bandwidth
    estimation for the `KDEMultivariate`, `KDEMultivariateConditional`,
    `KernelReg` and `CensoredKernelReg` classes behaves.

    Parameters
    ----------
    efficient : bool, optional
        If True, the bandwidth estimation is to be performed
        efficiently -- by taking smaller sub-samples and estimating
        the scaling factor of each subsample.  This is useful for large
        samples (nobs >> 300) and/or multiple variables (k_vars > 3).
        If False (default), all data is used at the same time.
    randomize : bool, optional
        If True, the bandwidth estimation is to be performed by
        taking `n_res` random resamples (with replacement) of size `n_sub` from
        the full sample.  If set to False (default), the estimation is
        performed by slicing the full sample in sub-samples of size `n_sub` so
        that all samples are used once.
    n_sub : int, optional
        Size of the sub-samples.  Default is 50.
    n_res : int, optional
        The number of random re-samples used to estimate the bandwidth.
        Only has an effect if ``randomize == True``.  Default value is 25.
    return_median : bool, optional
        If True (default), the estimator uses the median of all scaling factors
        for each sub-sample to estimate the bandwidth of the full sample.
        If False, the estimator uses the mean.
    return_only_bw : bool, optional
        If True, the estimator is to use the bandwidth and not the
        scaling factor.  This is *not* theoretically justified.
        Should be used only for experimenting.
    n_jobs : int, optional
        The number of jobs to use for parallel estimation with
        ``joblib.Parallel``.  Default is -1, meaning ``n_cores - 1``, with
        ``n_cores`` the number of available CPU cores.
        See the `joblib documentation
        <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_ for more details.

    Examples
    --------
    >>> settings = EstimatorSettings(randomize=True, n_jobs=3)
    >>> k_dens = KDEMultivariate(data, var_type, defaults=settings)
    """

    def __init__(self, efficient=False, randomize=False, n_res=25, n_sub=50, return_median=True, return_only_bw=False, n_jobs=-1):
        self.efficient = efficient
        self.randomize = randomize
        self.n_res = n_res
        self.n_sub = n_sub
        self.return_median = return_median
        self.return_only_bw = return_only_bw
        self.n_jobs = n_jobs