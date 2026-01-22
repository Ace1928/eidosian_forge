from statsmodels.compat.pandas import MONTH_END, QUARTER_END
from collections import OrderedDict
from warnings import warn
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import int_like
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.multivariate.pca import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace._quarterly_ar1 import QuarterlyAR1
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import string_like
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.statespace import mlemodel, initialization
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.base.data import PandasData
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.tableformatting import fmt_params
def _apply_factor_multiplicities(self, factors, factor_orders, factor_multiplicities):
    """
        Expand `factors` and `factor_orders` to account for factor multiplity.

        For example, if there is a `global` factor with multiplicity 2, then
        this method expands that into `global.1` and `global.2` in both the
        `factors` and `factor_orders` dictionaries.

        Parameters
        ----------
        factors : dict
            Dictionary of {endog_name: list of factor names}
        factor_orders : dict
            Dictionary of {tuple of factor names: factor order}
        factor_multiplicities : dict
            Dictionary of {factor name: factor multiplicity}

        Returns
        -------
        new_factors : dict
            Dictionary of {endog_name: list of factor names}, with factor names
            expanded to incorporate multiplicities.
        new_factors : dict
            Dictionary of {tuple of factor names: factor order}, with factor
            names in each tuple expanded to incorporate multiplicities.
        """
    new_factors = {}
    for endog_name, factors_list in factors.items():
        new_factor_list = []
        for factor_name in factors_list:
            n = factor_multiplicities.get(factor_name, 1)
            if n > 1:
                new_factor_list += [f'{factor_name}.{i + 1}' for i in range(n)]
            else:
                new_factor_list.append(factor_name)
        new_factors[endog_name] = new_factor_list
    new_factor_orders = {}
    for block, factor_order in factor_orders.items():
        if not isinstance(block, tuple):
            block = (block,)
        new_block = []
        for factor_name in block:
            n = factor_multiplicities.get(factor_name, 1)
            if n > 1:
                new_block += [f'{factor_name}.{i + 1}' for i in range(n)]
            else:
                new_block += [factor_name]
        new_factor_orders[tuple(new_block)] = factor_order
    return (new_factors, new_factor_orders)