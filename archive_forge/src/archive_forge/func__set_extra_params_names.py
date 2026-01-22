from __future__ import annotations
from statsmodels.compat.python import lzip
from functools import reduce
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import handle_data
from statsmodels.base.optimizer import Optimizer
import statsmodels.base.wrapper as wrap
from statsmodels.formula import handle_formula_data
from statsmodels.stats.contrast import (
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import (
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import nan_dot, recipr
from statsmodels.tools.validation import bool_like
def _set_extra_params_names(self, extra_params_names):
    if extra_params_names is not None:
        if self.exog is not None:
            self.exog_names.extend(extra_params_names)
        else:
            self.data.xnames = extra_params_names
        self.k_extra = len(extra_params_names)
        if hasattr(self, 'df_resid'):
            self.df_resid -= self.k_extra
    self.nparams = len(self.exog_names)