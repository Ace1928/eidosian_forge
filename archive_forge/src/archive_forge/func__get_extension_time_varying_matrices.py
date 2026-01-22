from warnings import warn
import numpy as np
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams
from statsmodels.tsa.tsatools import lagmat
from .initialization import Initialization
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .tools import (
def _get_extension_time_varying_matrices(self, params, exog, out_of_sample, extend_kwargs=None, transformed=True, includes_fixed=False, **kwargs):
    """
        Get time-varying state space system matrices for extended model

        Notes
        -----
        We need to override this method for SARIMAX because we need some
        special handling in the `simple_differencing=True` case.
        """
    exog = self._validate_out_of_sample_exog(exog, out_of_sample)
    if self.simple_differencing:
        nobs = self.data.orig_endog.shape[0] + out_of_sample
        tmp_endog = np.zeros((nobs, self.k_endog))
        if exog is not None:
            tmp_exog = np.c_[self.data.orig_exog.T, exog.T].T
        else:
            tmp_exog = None
    else:
        tmp_endog = np.zeros((out_of_sample, self.k_endog))
        tmp_exog = exog
    if extend_kwargs is None:
        extend_kwargs = {}
    if not self.simple_differencing and self.k_trend > 0:
        extend_kwargs.setdefault('trend_offset', self.trend_offset + self.nobs)
    extend_kwargs.setdefault('validate_specification', False)
    mod_extend = self.clone(endog=tmp_endog, exog=tmp_exog, **extend_kwargs)
    mod_extend.update(params, transformed=transformed, includes_fixed=includes_fixed)
    for name in self.ssm.shapes.keys():
        if name == 'obs' or name in kwargs:
            continue
        original = getattr(self.ssm, name)
        extended = getattr(mod_extend.ssm, name)
        so = original.shape[-1]
        se = extended.shape[-1]
        if (so > 1 or se > 1) or (so == 1 and self.nobs == 1 and np.any(original[..., 0] != extended[..., 0])):
            kwargs[name] = extended[..., -out_of_sample:]
    return kwargs