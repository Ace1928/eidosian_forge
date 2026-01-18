from pystatsmodels mailinglist 20100524
from collections import namedtuple
from statsmodels.compat.python import lzip, lrange
import copy
import math
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy import stats, interpolate
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.multitest import multipletests, _ecdf as ecdf, fdrcorrection as fdrcorrection0, fdrcorrection_twostage
from statsmodels.graphics import utils
from statsmodels.tools.sm_exceptions import ValueWarning
def tukey_pvalues(std_range, nm, df):
    contr = contrast_allpairs(nm)
    corr = np.dot(contr, contr.T) / 2.0
    tstat = std_range / np.sqrt(2) * np.ones(corr.shape[0])
    return multicontrast_pvalues(tstat, corr, df=df)