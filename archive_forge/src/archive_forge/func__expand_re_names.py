import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
def _expand_re_names(self, group_ix):
    names = list(self.model.data.exog_re_names)
    for j, v in enumerate(self.model.exog_vc.names):
        vg = self.model.exog_vc.colnames[j][group_ix]
        na = ['{}[{}]'.format(v, s) for s in vg]
        names.extend(na)
    return names