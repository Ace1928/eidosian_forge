from statsmodels.compat.pandas import is_int_index
import contextlib
import warnings
import datetime as dt
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tools.tools import pinv_extended, Bunch
from statsmodels.tools.sm_exceptions import PrecisionWarning, ValueWarning
from statsmodels.tools.numdiff import (_get_epsilon, approx_hess_cs,
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, aicc, bic, hqic
import statsmodels.base.wrapper as wrap
import statsmodels.tsa.base.prediction as pred
from statsmodels.base.data import PandasData
import statsmodels.tsa.base.tsa_model as tsbase
from .news import NewsResults
from .simulation_smoother import SimulationSmoother
from .kalman_smoother import SmootherResults
from .kalman_filter import INVERT_UNIVARIATE, SOLVE_LU, MEMORY_CONSERVE
from .initialization import Initialization
from .tools import prepare_exog, concat, _safe_cond, get_impact_dates
def _get_previous_updated(self, comparison, exog=None, comparison_type=None, **kwargs):
    comparison_dataset = not isinstance(comparison, (MLEResults, MLEResultsWrapper))
    if comparison_dataset:
        nobs_endog = len(comparison)
        nobs_exog = len(exog) if exog is not None else nobs_endog
        if nobs_exog > nobs_endog:
            _, _, _, ix = self.model._get_prediction_index(start=0, end=nobs_exog - 1)
            comparison = np.asarray(comparison)
            if comparison.ndim < 2:
                comparison = np.atleast_2d(comparison).T
            if comparison.ndim != 2 or comparison.shape[1] != self.model.k_endog:
                raise ValueError(f'Invalid shape for `comparison`. Must contain {self.model.k_endog} columns.')
            extra = np.zeros((nobs_exog - nobs_endog, self.model.k_endog)) * np.nan
            comparison = pd.DataFrame(np.concatenate([comparison, extra], axis=0), index=ix, columns=self.model.endog_names)
        comparison = self.apply(comparison, exog=exog, copy_initialization=True, **kwargs)
    nmissing = self.filter_results.missing.sum()
    nmissing_comparison = comparison.filter_results.missing.sum()
    if comparison_type == 'updated' or (comparison_type is None and (comparison.nobs > self.nobs or (comparison.nobs == self.nobs and nmissing > nmissing_comparison))):
        updated = comparison
        previous = self
    elif comparison_type == 'previous' or (comparison_type is None and (comparison.nobs < self.nobs or (comparison.nobs == self.nobs and nmissing < nmissing_comparison))):
        updated = self
        previous = comparison
    else:
        raise ValueError('Could not automatically determine the type of comparison requested to compute the News, so it must be specified as "updated" or "previous", using the `comparison_type` keyword argument')
    diff = previous.model._index.difference(updated.model._index)
    if len(diff) > 0:
        raise ValueError('The index associated with the updated results is not a superset of the index associated with the previous results, and so these datasets do not appear to be related. Can only compute the news by comparing this results set to previous results objects.')
    return (previous, updated, comparison_dataset)