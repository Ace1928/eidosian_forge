import itertools
import six
import numpy as np
from patsy import PatsyError
from patsy.categorical import (guess_categorical,
from patsy.util import (atleast_2d_column_default,
from patsy.design_info import (DesignMatrix, DesignInfo,
from patsy.redundancy import pick_contrasts_for_term
from patsy.eval import EvalEnvironment
from patsy.contrasts import code_contrast_matrix, Treatment
from patsy.compat import OrderedDict
from patsy.missing import NAAction
def _max_allowed_dim(dim, arr, factor):
    if arr.ndim > dim:
        msg = "factor '%s' evaluates to an %s-dimensional array; I only handle arrays with dimension <= %s" % (factor.name(), arr.ndim, dim)
        raise PatsyError(msg, factor)