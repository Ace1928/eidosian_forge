import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
def _perturb_bootstrap(self, vname):
    """
        Perturbs the model's parameters using a bootstrap.
        """
    endog, exog, init_kwds, fit_kwds = self.get_fitting_data(vname)
    m = len(endog)
    rix = np.random.randint(0, m, m)
    endog = endog[rix]
    exog = exog[rix, :]
    init_kwds = self._boot_kwds(init_kwds, rix)
    fit_kwds = self._boot_kwds(fit_kwds, rix)
    klass = self.model_class[vname]
    self.models[vname] = klass(endog, exog, **init_kwds)
    if vname in self.regularized and self.regularized[vname]:
        self.results[vname] = self.models[vname].fit_regularized(**fit_kwds)
    else:
        self.results[vname] = self.models[vname].fit(**fit_kwds)
    self.params[vname] = self.results[vname].params