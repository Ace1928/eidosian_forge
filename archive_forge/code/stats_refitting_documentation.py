import logging
import warnings
import numpy as np
from .stats import loo
from .stats_utils import logsumexp as _logsumexp
Recalculate exact Leave-One-Out cross validation refitting where the approximation fails.

    ``az.loo`` estimates the values of Leave-One-Out (LOO) cross validation using Pareto
    Smoothed Importance Sampling (PSIS) to approximate its value. PSIS works well when
    the posterior and the posterior_i (excluding observation i from the data used to fit)
    are similar. In some cases, there are highly influential observations for which PSIS
    cannot approximate the LOO-CV, and a warning of a large Pareto shape is sent by ArviZ.
    This cases typically have a handful of bad or very bad Pareto shapes and a majority of
    good or ok shapes.

    Therefore, this may not indicate that the model is not robust enough
    nor that these observations are inherently bad, only that PSIS cannot approximate LOO-CV
    correctly. Thus, we can use PSIS for all observations where the Pareto shape is below a
    threshold and refit the model to perform exact cross validation for the handful of
    observations where PSIS cannot be used. This approach allows to properly approximate
    LOO-CV with only a handful of refits, which in most cases is still much less computationally
    expensive than exact LOO-CV, which needs one refit per observation.

    Parameters
    ----------
    wrapper: SamplingWrapper-like
        Class (preferably a subclass of ``az.SamplingWrapper``, see :ref:`wrappers_api`
        for details) implementing the methods described
        in the SamplingWrapper docs. This allows ArviZ to call **any** sampling backend
        (like PyStan or emcee) using always the same syntax.
    loo_orig : ELPDData, optional
        ELPDData instance with pointwise loo results. The pareto_k attribute will be checked
        for values above the threshold.
    k_thresh : float, optional
        Pareto shape threshold. Each pareto shape value above ``k_thresh`` will trigger
        a refit excluding that observation.
    scale : str, optional
        Only taken into account when loo_orig is None. See ``az.loo`` for valid options.

    Returns
    -------
    ELPDData
        ELPDData instance containing the PSIS approximation where possible and the exact
        LOO-CV result where PSIS failed. The Pareto shape of the observations where exact
        LOO-CV was performed is artificially set to 0, but as PSIS is not performed, it
        should be ignored.

    Notes
    -----
    It is strongly recommended to first compute ``az.loo`` on the inference results to
    confirm that the number of values above the threshold is small enough. Otherwise,
    prohibitive computation time may be needed to perform all required refits.

    As an extreme case, artificially assigning all ``pareto_k`` values to something
    larger than the threshold would make ``reloo`` perform the whole exact LOO-CV.
    This is not generally recommended
    nor intended, however, if needed, this function can be used to achieve the result.

    Warnings
    --------
    Sampling wrappers are an experimental feature in a very early stage. Please use them
    with caution.
    