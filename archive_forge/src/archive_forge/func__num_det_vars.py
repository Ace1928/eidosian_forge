from collections import defaultdict
import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy
import scipy.stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.validation import string_like
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.coint_tables import c_sja, c_sjt
from statsmodels.tsa.tsatools import duplication_matrix, lagmat, vec
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
import statsmodels.tsa.vector_ar.irf as irf
import statsmodels.tsa.vector_ar.plotting as plot
from statsmodels.tsa.vector_ar.util import get_index, seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import (
def _num_det_vars(det_string, seasons=0):
    """Gives the number of deterministic variables specified by det_string and
    seasons.

    Parameters
    ----------
    det_string : str {"n", "co", "ci", "lo", "li"}
        * "n" - no deterministic terms
        * "co" - constant outside the cointegration relation
        * "ci" - constant within the cointegration relation
        * "lo" - linear trend outside the cointegration relation
        * "li" - linear trend within the cointegration relation

        Combinations of these are possible (e.g. "cili" or "colo" for linear
        trend with intercept). See the docstring of the :class:`VECM`-class for
        more information.
    seasons : int
        Number of periods in a seasonal cycle.

    Returns
    -------
    num : int
        Number of deterministic terms and number dummy variables for seasonal
        terms.
    """
    num = 0
    det_string = string_like(det_string, 'det_string')
    if 'ci' in det_string or 'co' in det_string:
        num += 1
    if 'li' in det_string or 'lo' in det_string:
        num += 1
    if seasons > 0:
        num += seasons - 1
    return num