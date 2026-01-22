from statsmodels.base.elastic_net import RegularizedResults
from statsmodels.stats.regularized_covariance import _calc_nodewise_row, \
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
import numpy as np
def _helper_fit_partition(self, pnum, endog, exog, fit_kwds, init_kwds_e={}):
    """handles the model fitting for each machine. NOTE: this
    is primarily handled outside of DistributedModel because
    joblib cannot handle class methods.

    Parameters
    ----------
    self : DistributedModel class instance
        An instance of DistributedModel.
    pnum : scalar
        index of current partition.
    endog : array_like
        endogenous data for current partition.
    exog : array_like
        exogenous data for current partition.
    fit_kwds : dict-like
        Keywords needed for the model fitting.
    init_kwds_e : dict-like
        Additional init_kwds to add for each partition.

    Returns
    -------
    estimation_method result.  For the default,
    _est_regularized_debiased, a tuple.
    """
    temp_init_kwds = self.init_kwds.copy()
    temp_init_kwds.update(init_kwds_e)
    model = self.model_class(endog, exog, **temp_init_kwds)
    results = self.estimation_method(model, pnum, self.partitions, fit_kwds=fit_kwds, **self.estimation_kwds)
    return results