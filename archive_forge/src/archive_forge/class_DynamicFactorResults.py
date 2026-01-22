import numpy as np
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .tools import (
from statsmodels.multivariate.pca import PCA
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
from statsmodels.compat.pandas import Appender
class DynamicFactorResults(MLEResults):
    """
    Class to hold results from fitting an DynamicFactor model.

    Parameters
    ----------
    model : DynamicFactor instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the DynamicFactor model
        instance.
    coefficient_matrices_var : ndarray
        Array containing autoregressive lag polynomial coefficient matrices,
        ordered from lowest degree to highest.

    See Also
    --------
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.mlemodel.MLEResults
    """

    def __init__(self, model, params, filter_results, cov_type=None, **kwargs):
        super().__init__(model, params, filter_results, cov_type, **kwargs)
        self.df_resid = np.inf
        self.specification = Bunch(**{'k_endog': self.model.k_endog, 'enforce_stationarity': self.model.enforce_stationarity, 'k_factors': self.model.k_factors, 'factor_order': self.model.factor_order, 'error_order': self.model.error_order, 'error_var': self.model.error_var, 'error_cov_type': self.model.error_cov_type, 'k_exog': self.model.k_exog})
        self.coefficient_matrices_var = None
        if self.model.factor_order > 0:
            ar_params = np.array(self.params[self.model._params_factor_transition])
            k_factors = self.model.k_factors
            factor_order = self.model.factor_order
            self.coefficient_matrices_var = ar_params.reshape(k_factors * factor_order, k_factors).T.reshape(k_factors, k_factors, factor_order).T
        self.coefficient_matrices_error = None
        if self.model.error_order > 0:
            ar_params = np.array(self.params[self.model._params_error_transition])
            k_endog = self.model.k_endog
            error_order = self.model.error_order
            if self.model.error_var:
                self.coefficient_matrices_error = ar_params.reshape(k_endog * error_order, k_endog).T.reshape(k_endog, k_endog, error_order).T
            else:
                mat = np.zeros((k_endog, k_endog * error_order))
                mat[self.model._idx_error_diag] = ar_params
                self.coefficient_matrices_error = mat.T.reshape(error_order, k_endog, k_endog)

    @property
    def factors(self):
        """
        Estimates of unobserved factors

        Returns
        -------
        out : Bunch
            Has the following attributes shown in Notes.

        Notes
        -----
        The output is a bunch of the following format:

        - `filtered`: a time series array with the filtered estimate of
          the component
        - `filtered_cov`: a time series array with the filtered estimate of
          the variance/covariance of the component
        - `smoothed`: a time series array with the smoothed estimate of
          the component
        - `smoothed_cov`: a time series array with the smoothed estimate of
          the variance/covariance of the component
        - `offset`: an integer giving the offset in the state vector where
          this component begins
        """
        out = None
        spec = self.specification
        if spec.k_factors > 0:
            offset = 0
            end = spec.k_factors
            res = self.filter_results
            out = Bunch(filtered=res.filtered_state[offset:end], filtered_cov=res.filtered_state_cov[offset:end, offset:end], smoothed=None, smoothed_cov=None, offset=offset)
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[offset:end]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[offset:end, offset:end]
        return out

    @cache_readonly
    def coefficients_of_determination(self):
        """
        Coefficients of determination (:math:`R^2`) from regressions of
        individual estimated factors on endogenous variables.

        Returns
        -------
        coefficients_of_determination : ndarray
            A `k_endog` x `k_factors` array, where
            `coefficients_of_determination[i, j]` represents the :math:`R^2`
            value from a regression of factor `j` and a constant on endogenous
            variable `i`.

        Notes
        -----
        Although it can be difficult to interpret the estimated factor loadings
        and factors, it is often helpful to use the coefficients of
        determination from univariate regressions to assess the importance of
        each factor in explaining the variation in each endogenous variable.

        In models with many variables and factors, this can sometimes lend
        interpretation to the factors (for example sometimes one factor will
        load primarily on real variables and another on nominal variables).

        See Also
        --------
        plot_coefficients_of_determination
        """
        from statsmodels.tools import add_constant
        spec = self.specification
        coefficients = np.zeros((spec.k_endog, spec.k_factors))
        which = 'filtered' if self.smoothed_state is None else 'smoothed'
        for i in range(spec.k_factors):
            exog = add_constant(self.factors[which][i])
            for j in range(spec.k_endog):
                endog = self.filter_results.endog[j]
                coefficients[j, i] = OLS(endog, exog).fit().rsquared
        return coefficients

    def plot_coefficients_of_determination(self, endog_labels=None, fig=None, figsize=None):
        """
        Plot the coefficients of determination

        Parameters
        ----------
        endog_labels : bool, optional
            Whether or not to label the endogenous variables along the x-axis
            of the plots. Default is to include labels if there are 5 or fewer
            endogenous variables.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----

        Produces a `k_factors` x 1 plot grid. The `i`th plot shows a bar plot
        of the coefficients of determination associated with factor `i`. The
        endogenous variables are arranged along the x-axis according to their
        position in the `endog` array.

        See Also
        --------
        coefficients_of_determination
        """
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        spec = self.specification
        if endog_labels is None:
            endog_labels = spec.k_endog <= 5
        coefficients_of_determination = self.coefficients_of_determination
        plot_idx = 1
        locations = np.arange(spec.k_endog)
        for coeffs in coefficients_of_determination.T:
            ax = fig.add_subplot(spec.k_factors, 1, plot_idx)
            ax.set_ylim((0, 1))
            ax.set(title='Factor %i' % plot_idx, ylabel='$R^2$')
            bars = ax.bar(locations, coeffs)
            if endog_labels:
                width = bars[0].get_width()
                ax.xaxis.set_ticks(locations + width / 2)
                ax.xaxis.set_ticklabels(self.model.endog_names)
            else:
                ax.set(xlabel='Endogenous variables')
                ax.xaxis.set_ticks([])
            plot_idx += 1
        return fig

    @Appender(MLEResults.summary.__doc__)
    def summary(self, alpha=0.05, start=None, separate_params=True):
        from statsmodels.iolib.summary import summary_params
        spec = self.specification
        model_name = []
        if spec.k_factors > 0:
            if spec.factor_order > 0:
                model_type = 'DynamicFactor(factors=%d, order=%d)' % (spec.k_factors, spec.factor_order)
            else:
                model_type = 'StaticFactor(factors=%d)' % spec.k_factors
            model_name.append(model_type)
            if spec.k_exog > 0:
                model_name.append('%d regressors' % spec.k_exog)
        else:
            model_name.append('SUR(%d regressors)' % spec.k_exog)
        if spec.error_order > 0:
            error_type = 'VAR' if spec.error_var else 'AR'
            model_name.append('%s(%d) errors' % (error_type, spec.error_order))
        summary = super().summary(alpha=alpha, start=start, model_name=model_name, display_params=not separate_params)
        if separate_params:
            indices = np.arange(len(self.params))

            def make_table(self, mask, title, strip_end=True):
                res = (self, self.params[mask], self.bse[mask], self.zvalues[mask], self.pvalues[mask], self.conf_int(alpha)[mask])
                param_names = ['.'.join(name.split('.')[:-1]) if strip_end else name for name in np.array(self.data.param_names)[mask].tolist()]
                return summary_params(res, yname=None, xname=param_names, alpha=alpha, use_t=False, title=title)
            k_endog = self.model.k_endog
            k_exog = self.model.k_exog
            k_factors = self.model.k_factors
            factor_order = self.model.factor_order
            _factor_order = self.model._factor_order
            _error_order = self.model._error_order
            loading_indices = indices[self.model._params_loadings]
            loading_masks = []
            exog_indices = indices[self.model._params_exog]
            exog_masks = []
            for i in range(k_endog):
                loading_mask = loading_indices[i * k_factors:(i + 1) * k_factors]
                loading_masks.append(loading_mask)
                exog_mask = exog_indices[i * k_exog:(i + 1) * k_exog]
                exog_masks.append(exog_mask)
                mask = np.concatenate([loading_mask, exog_mask])
                title = 'Results for equation %s' % self.model.endog_names[i]
                table = make_table(self, mask, title)
                summary.tables.append(table)
            factor_indices = indices[self.model._params_factor_transition]
            factor_masks = []
            if factor_order > 0:
                for i in range(k_factors):
                    start = i * _factor_order
                    factor_mask = factor_indices[start:start + _factor_order]
                    factor_masks.append(factor_mask)
                    title = 'Results for factor equation f%d' % (i + 1)
                    table = make_table(self, factor_mask, title)
                    summary.tables.append(table)
            error_masks = []
            if spec.error_order > 0:
                error_indices = indices[self.model._params_error_transition]
                for i in range(k_endog):
                    if spec.error_var:
                        start = i * _error_order
                        end = (i + 1) * _error_order
                    else:
                        start = i * spec.error_order
                        end = (i + 1) * spec.error_order
                    error_mask = error_indices[start:end]
                    error_masks.append(error_mask)
                    title = 'Results for error equation e(%s)' % self.model.endog_names[i]
                    table = make_table(self, error_mask, title)
                    summary.tables.append(table)
            error_cov_mask = indices[self.model._params_error_cov]
            table = make_table(self, error_cov_mask, 'Error covariance matrix', strip_end=False)
            summary.tables.append(table)
            masks = []
            for m in (loading_masks, exog_masks, factor_masks, error_masks, [error_cov_mask]):
                m = np.array(m).flatten()
                if len(m) > 0:
                    masks.append(m)
            masks = np.concatenate(masks)
            inverse_mask = np.array(list(set(indices).difference(set(masks))))
            if len(inverse_mask) > 0:
                table = make_table(self, inverse_mask, 'Other parameters', strip_end=False)
                summary.tables.append(table)
        return summary