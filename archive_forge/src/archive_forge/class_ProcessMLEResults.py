import numpy as np
import pandas as pd
import patsy
import statsmodels.base.model as base
from statsmodels.regression.linear_model import OLS
import collections
from scipy.optimize import minimize
from statsmodels.iolib import summary2
from statsmodels.tools.numdiff import approx_fprime
import warnings
class ProcessMLEResults(base.GenericLikelihoodModelResults):
    """
    Results class for Gaussian process regression models.
    """

    def __init__(self, model, mlefit):
        super().__init__(model, mlefit)
        pa = model.unpack(mlefit.params)
        self.mean_params = pa[0]
        self.scale_params = pa[1]
        self.smooth_params = pa[2]
        self.no_params = pa[3]
        self.df_resid = model.endog.shape[0] - len(mlefit.params)
        self.k_exog = self.model.exog.shape[1]
        self.k_scale = self.model.exog_scale.shape[1]
        self.k_smooth = self.model.exog_smooth.shape[1]
        self._has_noise = model._has_noise
        if model._has_noise:
            self.k_noise = self.model.exog_noise.shape[1]

    def predict(self, exog=None, transform=True, *args, **kwargs):
        if not transform:
            warnings.warn("'transform=False' is ignored in predict")
        if len(args) > 0 or len(kwargs) > 0:
            warnings.warn("extra arguments ignored in 'predict'")
        return self.model.predict(self.params, exog)

    def covariance(self, time, scale, smooth):
        """
        Returns a fitted covariance matrix.

        Parameters
        ----------
        time : array_like
            The time points at which the fitted covariance
            matrix is calculated.
        scale : array_like
            The data used to determine the scale parameter,
            must have len(time) rows.
        smooth : array_like
            The data used to determine the smoothness parameter,
            must have len(time) rows.

        Returns
        -------
        A covariance matrix.

        Notes
        -----
        If the model was fit using formulas, `scale` and `smooth` should
        be Dataframes, containing all variables that were present in the
        respective scaling and smoothing formulas used to fit the model.
        Otherwise, `scale` and `smooth` should be data arrays whose
        columns align with the fitted scaling and smoothing parameters.
        """
        return self.model.covariance(time, self.scale_params, self.smooth_params, scale, smooth)

    def covariance_group(self, group):
        ix = self.model._groups_ix[group]
        if len(ix) == 0:
            msg = "Group '%s' does not exist" % str(group)
            raise ValueError(msg)
        scale_data = self.model.exog_scale[ix, :]
        smooth_data = self.model.exog_smooth[ix, :]
        _, scale_names, smooth_names, _ = self.model._split_param_names()
        scale_data = pd.DataFrame(scale_data, columns=scale_names)
        smooth_data = pd.DataFrame(smooth_data, columns=smooth_names)
        time = self.model.time[ix]
        return self.model.covariance(time, self.scale_params, self.smooth_params, scale_data, smooth_data)

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        df = pd.DataFrame()
        typ = ['Mean'] * self.k_exog + ['Scale'] * self.k_scale + ['Smooth'] * self.k_smooth
        if self._has_noise:
            typ += ['SD'] * self.k_noise
        df['Type'] = typ
        df['coef'] = self.params
        try:
            df['std err'] = np.sqrt(np.diag(self.cov_params()))
        except Exception:
            df['std err'] = np.nan
        from scipy.stats.distributions import norm
        df['tvalues'] = df.coef / df['std err']
        df['P>|t|'] = 2 * norm.sf(np.abs(df.tvalues))
        f = norm.ppf(1 - alpha / 2)
        df['[%.3f' % (alpha / 2)] = df.coef - f * df['std err']
        df['%.3f]' % (1 - alpha / 2)] = df.coef + f * df['std err']
        df.index = self.model.data.param_names
        summ = summary2.Summary()
        if title is None:
            title = 'Gaussian process regression results'
        summ.add_title(title)
        summ.add_df(df)
        return summ