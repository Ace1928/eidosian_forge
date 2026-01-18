from statsmodels.compat.python import lzip
from statsmodels.compat.pandas import Appender
import numpy as np
from scipy import stats
import pandas as pd
import patsy
from collections import defaultdict
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults
from statsmodels.genmod import cov_struct as cov_structs
import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
import warnings
from statsmodels.graphics._regressionplots_doc import (
from statsmodels.discrete.discrete_margins import (
def update_cached_means(self, mean_params):
    """
        cached_means should always contain the most recent calculation
        of the group-wise mean vectors.  This function should be
        called every time the regression parameters are changed, to
        keep the cached means up to date.
        """
    endog = self.endog_li
    exog = self.exog_li
    offset = self.offset_li
    linkinv = self.family.link.inverse
    self.cached_means = []
    for i in range(self.num_group):
        if len(endog[i]) == 0:
            continue
        lpr = np.dot(exog[i], mean_params)
        if offset is not None:
            lpr += offset[i]
        expval = linkinv(lpr)
        self.cached_means.append((expval, lpr))