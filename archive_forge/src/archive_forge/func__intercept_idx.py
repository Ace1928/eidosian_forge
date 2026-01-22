import statsmodels.tools.data as data_util
from patsy import dmatrices, NAAction
import numpy as np
def _intercept_idx(design_info):
    """
    Returns boolean array index indicating which column holds the intercept.
    """
    from patsy.desc import INTERCEPT
    from numpy import array
    return array([INTERCEPT == i for i in design_info.terms])