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
class VARResultsWrapper(wrap.ResultsWrapper):
    _attrs = {'bse': 'columns_eq', 'cov_params': 'cov', 'params': 'columns_eq', 'pvalues': 'columns_eq', 'tvalues': 'columns_eq', 'sigma_u': 'cov_eq', 'sigma_u_mle': 'cov_eq', 'stderr': 'columns_eq'}
    _wrap_attrs = wrap.union_dicts(TimeSeriesResultsWrapper._wrap_attrs, _attrs)
    _methods = {'conf_int': 'multivariate_confint'}
    _wrap_methods = wrap.union_dicts(TimeSeriesResultsWrapper._wrap_methods, _methods)