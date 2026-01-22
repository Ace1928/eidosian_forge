import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
def _perturb_gaussian(self, vname):
    """
        Gaussian perturbation of model parameters.

        The normal approximation to the sampling distribution of the
        parameter estimates is used to define the mean and covariance
        structure of the perturbation distribution.
        """
    endog, exog, init_kwds, fit_kwds = self.get_fitting_data(vname)
    klass = self.model_class[vname]
    self.models[vname] = klass(endog, exog, **init_kwds)
    self.results[vname] = self.models[vname].fit(**fit_kwds)
    cov = self.results[vname].cov_params()
    mu = self.results[vname].params
    self.params[vname] = np.random.multivariate_normal(mean=mu, cov=cov)