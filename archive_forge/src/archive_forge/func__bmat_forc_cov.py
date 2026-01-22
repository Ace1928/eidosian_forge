from __future__ import annotations
from statsmodels.compat.python import lrange
from collections import defaultdict
from io import StringIO
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly, deprecated_alias
from statsmodels.tools.linalg import logdet_symm
from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tools.validation import array_like
from statsmodels.tsa.base.tsa_model import (
import statsmodels.tsa.tsatools as tsa
from statsmodels.tsa.tsatools import duplication_matrix, unvec, vec
from statsmodels.tsa.vector_ar import output, plotting, util
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.output import VARSummary
def _bmat_forc_cov(self):
    upper = np.zeros((self.k_exog, self.df_model))
    upper[:, :self.k_exog] = np.eye(self.k_exog)
    lower_dim = self.neqs * (self.k_ar - 1)
    eye = np.eye(lower_dim)
    lower = np.column_stack((np.zeros((lower_dim, self.k_exog)), eye, np.zeros((lower_dim, self.neqs))))
    return np.vstack((upper, self.params.T, lower))