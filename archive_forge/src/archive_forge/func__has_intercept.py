import statsmodels.tools.data as data_util
from patsy import dmatrices, NAAction
import numpy as np
def _has_intercept(design_info):
    from patsy.desc import INTERCEPT
    return INTERCEPT in design_info.terms