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
def _convert_vc(exog_vc):
    vc_names = []
    vc_colnames = []
    vc_mats = []
    groups = set()
    for k, v in exog_vc.items():
        groups |= set(v.keys())
    groups = list(groups)
    groups.sort()
    for k, v in exog_vc.items():
        vc_names.append(k)
        colnames, mats = ([], [])
        for g in groups:
            try:
                colnames.append(v[g].columns)
            except AttributeError:
                colnames.append([str(j) for j in range(v[g].shape[1])])
            mats.append(v[g])
        vc_colnames.append(colnames)
        vc_mats.append(mats)
    ii = np.argsort(vc_names)
    vc_names = [vc_names[i] for i in ii]
    vc_colnames = [vc_colnames[i] for i in ii]
    vc_mats = [vc_mats[i] for i in ii]
    return VCSpec(vc_names, vc_colnames, vc_mats)