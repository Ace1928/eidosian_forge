import warnings
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lzip
from collections import defaultdict
import numpy as np
from statsmodels.graphics._regressionplots_doc import _plot_influence_doc
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import maybe_unwrap_results
def _get_drop_vari(self, attributes):
    """
        regress endog on exog without one of the variables

        This uses a k_vars loop, only attributes of the OLS instance are
        stored.

        Parameters
        ----------
        attributes : list[str]
           These are the names of the attributes of the auxiliary OLS results
           instance that are stored and returned.

        not yet used
        """
    from statsmodels.sandbox.tools.cross_val import LeaveOneOut
    endog = self.results.model.endog
    exog = self.exog
    cv_iter = LeaveOneOut(self.k_vars)
    res_loo = defaultdict(list)
    for inidx, outidx in cv_iter:
        for att in attributes:
            res_i = self.model_class(endog, exog[:, inidx]).fit()
            res_loo[att].append(getattr(res_i, att))
    return res_loo