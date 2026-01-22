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
class DynamicFactorMQResults(mlemodel.MLEResults):
    """
    Results from fitting a dynamic factor model
    """

    def __init__(self, model, params, filter_results, cov_type=None, **kwargs):
        super().__init__(model, params, filter_results, cov_type, **kwargs)

    @property
    def factors(self):
        """
        Estimates of unobserved factors.

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
        if self.model.k_factors > 0:
            iloc = self.model._s.factors_L1
            ix = np.array(self.model.state_names)[iloc].tolist()
            out = Bunch(filtered=self.states.filtered.loc[:, ix], filtered_cov=self.states.filtered_cov.loc[np.s_[ix, :], ix], smoothed=None, smoothed_cov=None)
            if self.smoothed_state is not None:
                out.smoothed = self.states.smoothed.loc[:, ix]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.states.smoothed_cov.loc[np.s_[ix, :], ix]
        return out

    def get_coefficients_of_determination(self, method='individual', which=None):
        """
        Get coefficients of determination (R-squared) for variables / factors.

        Parameters
        ----------
        method : {'individual', 'joint', 'cumulative'}, optional
            The type of R-squared values to generate. "individual" plots
            the R-squared of each variable on each factor; "joint" plots the
            R-squared of each variable on each factor that it loads on;
            "cumulative" plots the successive R-squared values as each
            additional factor is added to the regression, for each variable.
            Default is 'individual'.
        which: {None, 'filtered', 'smoothed'}, optional
            Whether to compute R-squared values based on filtered or smoothed
            estimates of the factors. Default is 'smoothed' if smoothed results
            are available and 'filtered' otherwise.

        Returns
        -------
        rsquared : pd.DataFrame or pd.Series
            The R-squared values from regressions of observed variables on
            one or more of the factors. If method='individual' or
            method='cumulative', this will be a Pandas DataFrame with observed
            variables as the index and factors as the columns . If
            method='joint', will be a Pandas Series with observed variables as
            the index.

        See Also
        --------
        plot_coefficients_of_determination
        coefficients_of_determination
        """
        from statsmodels.tools import add_constant
        method = string_like(method, 'method', options=['individual', 'joint', 'cumulative'])
        if which is None:
            which = 'filtered' if self.smoothed_state is None else 'smoothed'
        k_endog = self.model.k_endog
        k_factors = self.model.k_factors
        ef_map = self.model._s.endog_factor_map
        endog_names = self.model.endog_names
        factor_names = self.model.factor_names
        if method == 'individual':
            coefficients = np.zeros((k_endog, k_factors))
            for i in range(k_factors):
                exog = add_constant(self.factors[which].iloc[:, i])
                for j in range(k_endog):
                    if ef_map.iloc[j, i]:
                        endog = self.filter_results.endog[j]
                        coefficients[j, i] = OLS(endog, exog, missing='drop').fit().rsquared
                    else:
                        coefficients[j, i] = np.nan
            coefficients = pd.DataFrame(coefficients, index=endog_names, columns=factor_names)
        elif method == 'joint':
            coefficients = np.zeros((k_endog,))
            exog = add_constant(self.factors[which])
            for j in range(k_endog):
                endog = self.filter_results.endog[j]
                ix = np.r_[True, ef_map.iloc[j]].tolist()
                X = exog.loc[:, ix]
                coefficients[j] = OLS(endog, X, missing='drop').fit().rsquared
            coefficients = pd.Series(coefficients, index=endog_names)
        elif method == 'cumulative':
            coefficients = np.zeros((k_endog, k_factors))
            exog = add_constant(self.factors[which])
            for j in range(k_endog):
                endog = self.filter_results.endog[j]
                for i in range(k_factors):
                    if self.model._s.endog_factor_map.iloc[j, i]:
                        ix = np.r_[True, ef_map.iloc[j, :i + 1], [False] * (k_factors - i - 1)]
                        X = exog.loc[:, ix.astype(bool).tolist()]
                        coefficients[j, i] = OLS(endog, X, missing='drop').fit().rsquared
                    else:
                        coefficients[j, i] = np.nan
            coefficients = pd.DataFrame(coefficients, index=endog_names, columns=factor_names)
        return coefficients

    @cache_readonly
    def coefficients_of_determination(self):
        """
        Individual coefficients of determination (:math:`R^2`).

        Coefficients of determination (:math:`R^2`) from regressions of
        endogenous variables on individual estimated factors.

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
        get_coefficients_of_determination
        plot_coefficients_of_determination
        """
        return self.get_coefficients_of_determination(method='individual')

    def plot_coefficients_of_determination(self, method='individual', which=None, endog_labels=None, fig=None, figsize=None):
        """
        Plot coefficients of determination (R-squared) for variables / factors.

        Parameters
        ----------
        method : {'individual', 'joint', 'cumulative'}, optional
            The type of R-squared values to generate. "individual" plots
            the R-squared of each variable on each factor; "joint" plots the
            R-squared of each variable on each factor that it loads on;
            "cumulative" plots the successive R-squared values as each
            additional factor is added to the regression, for each variable.
            Default is 'individual'.
        which: {None, 'filtered', 'smoothed'}, optional
            Whether to compute R-squared values based on filtered or smoothed
            estimates of the factors. Default is 'smoothed' if smoothed results
            are available and 'filtered' otherwise.
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
        The endogenous variables are arranged along the x-axis according to
        their position in the model's `endog` array.

        See Also
        --------
        get_coefficients_of_determination
        """
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        method = string_like(method, 'method', options=['individual', 'joint', 'cumulative'])
        if endog_labels is None:
            endog_labels = self.model.k_endog <= 5
        rsquared = self.get_coefficients_of_determination(method=method, which=which)
        if method in ['individual', 'cumulative']:
            plot_idx = 1
            for factor_name, coeffs in rsquared.T.iterrows():
                ax = fig.add_subplot(self.model.k_factors, 1, plot_idx)
                ax.set_ylim((0, 1))
                ax.set(title=f'{factor_name}', ylabel='$R^2$')
                coeffs.plot(ax=ax, kind='bar')
                if plot_idx < len(rsquared.columns) or not endog_labels:
                    ax.xaxis.set_ticklabels([])
                plot_idx += 1
        elif method == 'joint':
            ax = fig.add_subplot(1, 1, 1)
            ax.set_ylim((0, 1))
            ax.set(title='$R^2$ - regression on all loaded factors', ylabel='$R^2$')
            rsquared.plot(ax=ax, kind='bar')
            if not endog_labels:
                ax.xaxis.set_ticklabels([])
        return fig

    def get_prediction(self, start=None, end=None, dynamic=False, information_set='predicted', signal_only=False, original_scale=True, index=None, exog=None, extend_model=None, extend_kwargs=None, **kwargs):
        """
        In-sample prediction and out-of-sample forecasting.

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.
        dynamic : bool, int, str, or datetime, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Can also be an absolute date string to parse or a
            datetime type (these are not interpreted as offsets).
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.
        information_set : str, optional
            The information set to condition each prediction on. Default is
            "predicted", which computes predictions of period t values
            conditional on observed data through period t-1; these are
            one-step-ahead predictions, and correspond with the typical
            `fittedvalues` results attribute. Alternatives are "filtered",
            which computes predictions of period t values conditional on
            observed data through period t, and "smoothed", which computes
            predictions of period t values conditional on the entire dataset
            (including also future observations t+1, t+2, ...).
        signal_only : bool, optional
            Whether to compute forecasts of only the "signal" component of
            the observation equation. Default is False. For example, the
            observation equation of a time-invariant model is
            :math:`y_t = d + Z \\alpha_t + \\varepsilon_t`, and the "signal"
            component is then :math:`Z \\alpha_t`. If this argument is set to
            True, then forecasts of the "signal" :math:`Z \\alpha_t` will be
            returned. Otherwise, the default is for forecasts of :math:`y_t`
            to be returned.
        original_scale : bool, optional
            If the model specification standardized the data, whether or not
            to return predictions in the original scale of the data (i.e.
            before it was standardized by the model). Default is True.
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. See `FilterResults.predict` for more details.

        Returns
        -------
        forecast : ndarray
            Array of out of in-sample predictions and / or out-of-sample
            forecasts. An (npredict x k_endog) array.
        """
        res = super().get_prediction(start=start, end=end, dynamic=dynamic, information_set=information_set, signal_only=signal_only, index=index, exog=exog, extend_model=extend_model, extend_kwargs=extend_kwargs, **kwargs)
        if self.model.standardize and original_scale:
            prediction_results = res.prediction_results
            k_endog, _ = prediction_results.endog.shape
            mean = np.array(self.model._endog_mean)
            std = np.array(self.model._endog_std)
            if self.model.k_endog > 1:
                mean = mean[None, :]
                std = std[None, :]
            res._results._predicted_mean = res._results._predicted_mean * std + mean
            if k_endog == 1:
                res._results._var_pred_mean *= std ** 2
            else:
                res._results._var_pred_mean = std * res._results._var_pred_mean * std.T
        return res

    def news(self, comparison, impact_date=None, impacted_variable=None, start=None, end=None, periods=None, exog=None, comparison_type=None, revisions_details_start=False, state_index=None, return_raw=False, tolerance=1e-10, endog_quarterly=None, original_scale=True, **kwargs):
        """
        Compute impacts from updated data (news and revisions).

        Parameters
        ----------
        comparison : array_like or MLEResults
            An updated dataset with updated and/or revised data from which the
            news can be computed, or an updated or previous results object
            to use in computing the news.
        impact_date : int, str, or datetime, optional
            A single specific period of impacts from news and revisions to
            compute. Can also be a date string to parse or a datetime type.
            This argument cannot be used in combination with `start`, `end`, or
            `periods`. Default is the first out-of-sample observation.
        impacted_variable : str, list, array, or slice, optional
            Observation variable label or slice of labels specifying that only
            specific impacted variables should be shown in the News output. The
            impacted variable(s) describe the variables that were *affected* by
            the news. If you do not know the labels for the variables, check
            the `endog_names` attribute of the model instance.
        start : int, str, or datetime, optional
            The first period of impacts from news and revisions to compute.
            Can also be a date string to parse or a datetime type. Default is
            the first out-of-sample observation.
        end : int, str, or datetime, optional
            The last period of impacts from news and revisions to compute.
            Can also be a date string to parse or a datetime type. Default is
            the first out-of-sample observation.
        periods : int, optional
            The number of periods of impacts from news and revisions to
            compute.
        exog : array_like, optional
            Array of exogenous regressors for the out-of-sample period, if
            applicable.
        comparison_type : {None, 'previous', 'updated'}
            This denotes whether the `comparison` argument represents a
            *previous* results object or dataset or an *updated* results object
            or dataset. If not specified, then an attempt is made to determine
            the comparison type.
        state_index : array_like or "common", optional
            An optional index specifying a subset of states to use when
            constructing the impacts of revisions and news. For example, if
            `state_index=[0, 1]` is passed, then only the impacts to the
            observed variables arising from the impacts to the first two
            states will be returned. If the string "common" is passed and the
            model includes idiosyncratic AR(1) components, news will only be
            computed based on the common states. Default is to use all states.
        return_raw : bool, optional
            Whether or not to return only the specific output or a full
            results object. Default is to return a full results object.
        tolerance : float, optional
            The numerical threshold for determining zero impact. Default is
            that any impact less than 1e-10 is assumed to be zero.
        endog_quarterly : array_like, optional
            New observations of quarterly variables, if `comparison` was
            provided as an updated monthly dataset. If this argument is
            provided, it must be a Pandas Series or DataFrame with a
            DatetimeIndex or PeriodIndex at the quarterly frequency.

        References
        ----------
        .. [1] Bańbura, Marta, and Michele Modugno.
               "Maximum likelihood estimation of factor models on datasets with
               arbitrary pattern of missing data."
               Journal of Applied Econometrics 29, no. 1 (2014): 133-160.
        .. [2] Bańbura, Marta, Domenico Giannone, and Lucrezia Reichlin.
               "Nowcasting."
               The Oxford Handbook of Economic Forecasting. July 8, 2011.
        .. [3] Bańbura, Marta, Domenico Giannone, Michele Modugno, and Lucrezia
               Reichlin.
               "Now-casting and the real-time data flow."
               In Handbook of economic forecasting, vol. 2, pp. 195-237.
               Elsevier, 2013.
        """
        if state_index == 'common':
            state_index = np.arange(self.model.k_states - self.model.k_endog)
        news_results = super().news(comparison, impact_date=impact_date, impacted_variable=impacted_variable, start=start, end=end, periods=periods, exog=exog, comparison_type=comparison_type, revisions_details_start=revisions_details_start, state_index=state_index, return_raw=return_raw, tolerance=tolerance, endog_quarterly=endog_quarterly, **kwargs)
        if not return_raw and self.model.standardize and original_scale:
            endog_mean = self.model._endog_mean
            endog_std = self.model._endog_std
            news_results.total_impacts = news_results.total_impacts * endog_std
            news_results.update_impacts = news_results.update_impacts * endog_std
            if news_results.revision_impacts is not None:
                news_results.revision_impacts = news_results.revision_impacts * endog_std
            if news_results.revision_detailed_impacts is not None:
                news_results.revision_detailed_impacts = news_results.revision_detailed_impacts * endog_std
            if news_results.revision_grouped_impacts is not None:
                news_results.revision_grouped_impacts = news_results.revision_grouped_impacts * endog_std
            for name in ['prev_impacted_forecasts', 'news', 'revisions', 'update_realized', 'update_forecasts', 'revised', 'revised_prev', 'post_impacted_forecasts', 'revisions_all', 'revised_all', 'revised_prev_all']:
                dta = getattr(news_results, name)
                orig_name = None
                if hasattr(dta, 'name'):
                    orig_name = dta.name
                dta = dta.multiply(endog_std, level=1)
                if name not in ['news', 'revisions']:
                    dta = dta.add(endog_mean, level=1)
                if orig_name is not None:
                    dta.name = orig_name
                setattr(news_results, name, dta)
            news_results.weights = news_results.weights.divide(endog_std, axis=0, level=1).multiply(endog_std, axis=1, level=1)
            news_results.revision_weights = news_results.revision_weights.divide(endog_std, axis=0, level=1).multiply(endog_std, axis=1, level=1)
        return news_results

    def get_smoothed_decomposition(self, decomposition_of='smoothed_state', state_index=None, original_scale=True):
        """
        Decompose smoothed output into contributions from observations

        Parameters
        ----------
        decomposition_of : {"smoothed_state", "smoothed_signal"}
            The object to perform a decomposition of. If it is set to
            "smoothed_state", then the elements of the smoothed state vector
            are decomposed into the contributions of each observation. If it
            is set to "smoothed_signal", then the predictions of the
            observation vector based on the smoothed state vector are
            decomposed. Default is "smoothed_state".
        state_index : array_like, optional
            An optional index specifying a subset of states to use when
            constructing the decomposition of the "smoothed_signal". For
            example, if `state_index=[0, 1]` is passed, then only the
            contributions of observed variables to the smoothed signal arising
            from the first two states will be returned. Note that if not all
            states are used, the contributions will not sum to the smoothed
            signal. Default is to use all states.
        original_scale : bool, optional
            If the model specification standardized the data, whether or not
            to return simulations in the original scale of the data (i.e.
            before it was standardized by the model). Default is True.

        Returns
        -------
        data_contributions : pd.DataFrame
            Contributions of observations to the decomposed object. If the
            smoothed state is being decomposed, then `data_contributions` is
            shaped `(k_states x nobs, k_endog x nobs)` with a `pd.MultiIndex`
            index corresponding to `state_to x date_to` and `pd.MultiIndex`
            columns corresponding to `variable_from x date_from`. If the
            smoothed signal is being decomposed, then `data_contributions` is
            shaped `(k_endog x nobs, k_endog x nobs)` with `pd.MultiIndex`-es
            corresponding to `variable_to x date_to` and
            `variable_from x date_from`.
        obs_intercept_contributions : pd.DataFrame
            Contributions of the observation intercept to the decomposed
            object. If the smoothed state is being decomposed, then
            `obs_intercept_contributions` is
            shaped `(k_states x nobs, k_endog x nobs)` with a `pd.MultiIndex`
            index corresponding to `state_to x date_to` and `pd.MultiIndex`
            columns corresponding to `obs_intercept_from x date_from`. If the
            smoothed signal is being decomposed, then
            `obs_intercept_contributions` is shaped
            `(k_endog x nobs, k_endog x nobs)` with `pd.MultiIndex`-es
            corresponding to `variable_to x date_to` and
            `obs_intercept_from x date_from`.
        state_intercept_contributions : pd.DataFrame
            Contributions of the state intercept to the decomposed
            object. If the smoothed state is being decomposed, then
            `state_intercept_contributions` is
            shaped `(k_states x nobs, k_states x nobs)` with a `pd.MultiIndex`
            index corresponding to `state_to x date_to` and `pd.MultiIndex`
            columns corresponding to `state_intercept_from x date_from`. If the
            smoothed signal is being decomposed, then
            `state_intercept_contributions` is shaped
            `(k_endog x nobs, k_states x nobs)` with `pd.MultiIndex`-es
            corresponding to `variable_to x date_to` and
            `state_intercept_from x date_from`.
        prior_contributions : pd.DataFrame
            Contributions of the prior to the decomposed object. If the
            smoothed state is being decomposed, then `prior_contributions` is
            shaped `(nobs x k_states, k_states)`, with a `pd.MultiIndex`
            index corresponding to `state_to x date_to` and columns
            corresponding to elements of the prior mean (aka "initial state").
            If the smoothed signal is being decomposed, then
            `prior_contributions` is shaped `(nobs x k_endog, k_states)`,
            with a `pd.MultiIndex` index corresponding to
            `variable_to x date_to` and columns corresponding to elements of
            the prior mean.

        Notes
        -----
        Denote the smoothed state at time :math:`t` by :math:`\\alpha_t`. Then
        the smoothed signal is :math:`Z_t \\alpha_t`, where :math:`Z_t` is the
        design matrix operative at time :math:`t`.
        """
        if self.model.standardize and original_scale:
            cache_obs_intercept = self.model['obs_intercept']
            self.model['obs_intercept'] = self.model._endog_mean
        data_contributions, obs_intercept_contributions, state_intercept_contributions, prior_contributions = super().get_smoothed_decomposition(decomposition_of=decomposition_of, state_index=state_index)
        if self.model.standardize and original_scale:
            self.model['obs_intercept'] = cache_obs_intercept
        if decomposition_of == 'smoothed_signal' and self.model.standardize and original_scale:
            endog_std = self.model._endog_std
            data_contributions = data_contributions.multiply(endog_std, axis=0, level=0)
            obs_intercept_contributions = obs_intercept_contributions.multiply(endog_std, axis=0, level=0)
            state_intercept_contributions = state_intercept_contributions.multiply(endog_std, axis=0, level=0)
            prior_contributions = prior_contributions.multiply(endog_std, axis=0, level=0)
        return (data_contributions, obs_intercept_contributions, state_intercept_contributions, prior_contributions)

    def append(self, endog, endog_quarterly=None, refit=False, fit_kwargs=None, copy_initialization=True, retain_standardization=True, **kwargs):
        """
        Recreate the results object with new data appended to original data.

        Creates a new result object applied to a dataset that is created by
        appending new data to the end of the model's original data. The new
        results can then be used for analysis or forecasting.

        Parameters
        ----------
        endog : array_like
            New observations from the modeled time-series process.
        endog_quarterly : array_like, optional
            New observations of quarterly variables. If provided, must be a
            Pandas Series or DataFrame with a DatetimeIndex or PeriodIndex at
            the quarterly frequency.
        refit : bool, optional
            Whether to re-fit the parameters, based on the combined dataset.
            Default is False (so parameters from the current results object
            are used to create the new results object).
        fit_kwargs : dict, optional
            Keyword arguments to pass to `fit` (if `refit=True`) or `filter` /
            `smooth`.
        copy_initialization : bool, optional
            Whether or not to copy the initialization from the current results
            set to the new model. Default is True.
        retain_standardization : bool, optional
            Whether or not to use the mean and standard deviations that were
            used to standardize the data in the current model in the new model.
            Default is True.
        **kwargs
            Keyword arguments may be used to modify model specification
            arguments when created the new model object.

        Returns
        -------
        results
            Updated Results object, that includes results from both the
            original dataset and the new dataset.

        Notes
        -----
        The `endog` and `exog` arguments to this method must be formatted in
        the same way (e.g. Pandas Series versus Numpy array) as were the
        `endog` and `exog` arrays passed to the original model.

        The `endog` (and, if applicable, `endog_quarterly`) arguments to this
        method should consist of new observations that occurred directly after
        the last element of `endog`. For any other kind of dataset, see the
        `apply` method.

        This method will apply filtering to all of the original data as well
        as to the new data. To apply filtering only to the new data (which
        can be much faster if the original dataset is large), see the `extend`
        method.

        See Also
        --------
        extend
        apply
        """
        endog, k_endog_monthly = DynamicFactorMQ.construct_endog(endog, endog_quarterly)
        k_endog = endog.shape[1] if len(endog.shape) == 2 else 1
        if k_endog_monthly != self.model.k_endog_M or k_endog != self.model.k_endog:
            raise ValueError('Cannot append data of a different dimension to a model.')
        kwargs['k_endog_monthly'] = k_endog_monthly
        return super().append(endog, refit=refit, fit_kwargs=fit_kwargs, copy_initialization=copy_initialization, retain_standardization=retain_standardization, **kwargs)

    def extend(self, endog, endog_quarterly=None, fit_kwargs=None, retain_standardization=True, **kwargs):
        """
        Recreate the results object for new data that extends original data.

        Creates a new result object applied to a new dataset that is assumed to
        follow directly from the end of the model's original data. The new
        results can then be used for analysis or forecasting.

        Parameters
        ----------
        endog : array_like
            New observations from the modeled time-series process.
        endog_quarterly : array_like, optional
            New observations of quarterly variables. If provided, must be a
            Pandas Series or DataFrame with a DatetimeIndex or PeriodIndex at
            the quarterly frequency.
        fit_kwargs : dict, optional
            Keyword arguments to pass to `filter` or `smooth`.
        retain_standardization : bool, optional
            Whether or not to use the mean and standard deviations that were
            used to standardize the data in the current model in the new model.
            Default is True.
        **kwargs
            Keyword arguments may be used to modify model specification
            arguments when created the new model object.

        Returns
        -------
        results
            Updated Results object, that includes results only for the new
            dataset.

        See Also
        --------
        append
        apply

        Notes
        -----
        The `endog` argument to this method should consist of new observations
        that occurred directly after the last element of the model's original
        `endog` array. For any other kind of dataset, see the `apply` method.

        This method will apply filtering only to the new data provided by the
        `endog` argument, which can be much faster than re-filtering the entire
        dataset. However, the returned results object will only have results
        for the new data. To retrieve results for both the new data and the
        original data, see the `append` method.
        """
        endog, k_endog_monthly = DynamicFactorMQ.construct_endog(endog, endog_quarterly)
        k_endog = endog.shape[1] if len(endog.shape) == 2 else 1
        if k_endog_monthly != self.model.k_endog_M or k_endog != self.model.k_endog:
            raise ValueError('Cannot append data of a different dimension to a model.')
        kwargs['k_endog_monthly'] = k_endog_monthly
        return super().extend(endog, fit_kwargs=fit_kwargs, retain_standardization=retain_standardization, **kwargs)

    def apply(self, endog, k_endog_monthly=None, endog_quarterly=None, refit=False, fit_kwargs=None, copy_initialization=False, retain_standardization=True, **kwargs):
        """
        Apply the fitted parameters to new data unrelated to the original data.

        Creates a new result object using the current fitted parameters,
        applied to a completely new dataset that is assumed to be unrelated to
        the model's original data. The new results can then be used for
        analysis or forecasting.

        Parameters
        ----------
        endog : array_like
            New observations from the modeled time-series process.
        k_endog_monthly : int, optional
            If specifying a monthly/quarterly mixed frequency model in which
            the provided `endog` dataset contains both the monthly and
            quarterly data, this variable should be used to indicate how many
            of the variables are monthly.
        endog_quarterly : array_like, optional
            New observations of quarterly variables. If provided, must be a
            Pandas Series or DataFrame with a DatetimeIndex or PeriodIndex at
            the quarterly frequency.
        refit : bool, optional
            Whether to re-fit the parameters, using the new dataset.
            Default is False (so parameters from the current results object
            are used to create the new results object).
        fit_kwargs : dict, optional
            Keyword arguments to pass to `fit` (if `refit=True`) or `filter` /
            `smooth`.
        copy_initialization : bool, optional
            Whether or not to copy the initialization from the current results
            set to the new model. Default is False.
        retain_standardization : bool, optional
            Whether or not to use the mean and standard deviations that were
            used to standardize the data in the current model in the new model.
            Default is True.
        **kwargs
            Keyword arguments may be used to modify model specification
            arguments when created the new model object.

        Returns
        -------
        results
            Updated Results object, that includes results only for the new
            dataset.

        See Also
        --------
        statsmodels.tsa.statespace.mlemodel.MLEResults.append
        statsmodels.tsa.statespace.mlemodel.MLEResults.apply

        Notes
        -----
        The `endog` argument to this method should consist of new observations
        that are not necessarily related to the original model's `endog`
        dataset. For observations that continue that original dataset by follow
        directly after its last element, see the `append` and `extend` methods.
        """
        mod = self.model.clone(endog, k_endog_monthly=k_endog_monthly, endog_quarterly=endog_quarterly, retain_standardization=retain_standardization, **kwargs)
        if copy_initialization:
            init = initialization.Initialization.from_results(self.filter_results)
            mod.ssm.initialization = init
        res = self._apply(mod, refit=refit, fit_kwargs=fit_kwargs)
        return res

    def summary(self, alpha=0.05, start=None, title=None, model_name=None, display_params=True, display_diagnostics=False, display_params_as_list=False, truncate_endog_names=None, display_max_endog=3):
        """
        Summarize the Model.

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals. Default is 0.05.
        start : int, optional
            Integer of the start observation. Default is 0.
        title : str, optional
            The title used for the summary table.
        model_name : str, optional
            The name of the model used. Default is to use model class name.

        Returns
        -------
        summary : Summary instance
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        mod = self.model
        if title is None:
            title = 'Dynamic Factor Results'
        if model_name is None:
            model_name = self.model._model_name
        endog_names = self.model._get_endog_names(truncate=truncate_endog_names)
        extra_top_left = None
        extra_top_right = []
        mle_retvals = getattr(self, 'mle_retvals', None)
        mle_settings = getattr(self, 'mle_settings', None)
        if mle_settings is not None and mle_settings.method == 'em':
            extra_top_right += [('EM Iterations', [f'{mle_retvals.iter}'])]
        summary = super().summary(alpha=alpha, start=start, title=title, model_name=model_name, display_params=display_params and display_params_as_list, display_diagnostics=display_diagnostics, truncate_endog_names=truncate_endog_names, display_max_endog=display_max_endog, extra_top_left=extra_top_left, extra_top_right=extra_top_right)
        table_ix = 1
        if not display_params_as_list:
            data = pd.DataFrame(self.filter_results.design[:, mod._s['factors_L1'], 0], index=endog_names, columns=mod.factor_names)
            try:
                data = data.map(lambda s: '%.2f' % s)
            except AttributeError:
                data = data.applymap(lambda s: '%.2f' % s)
            k_idio = 1
            if mod.idiosyncratic_ar1:
                data['   idiosyncratic: AR(1)'] = self.params[mod._p['idiosyncratic_ar1']]
                k_idio += 1
            data['var.'] = self.params[mod._p['idiosyncratic_var']]
            cols_to_cast = data.columns[-k_idio:]
            data[cols_to_cast] = data[cols_to_cast].astype(object)
            try:
                data.iloc[:, -k_idio:] = data.iloc[:, -k_idio:].map(lambda s: f'{s:.2f}')
            except AttributeError:
                data.iloc[:, -k_idio:] = data.iloc[:, -k_idio:].applymap(lambda s: f'{s:.2f}')
            data.index.name = 'Factor loadings:'
            base_iloc = np.arange(mod.k_factors)
            for i in range(mod.k_endog):
                iloc = [j for j in base_iloc if j not in mod._s.endog_factor_iloc[i]]
                data.iloc[i, iloc] = '.'
            data = data.reset_index()
            params_data = data.values
            params_header = data.columns.tolist()
            params_stubs = None
            title = 'Observation equation:'
            table = SimpleTable(params_data, params_header, params_stubs, txt_fmt=fmt_params, title=title)
            summary.tables.insert(table_ix, table)
            table_ix += 1
            ix1 = 0
            ix2 = 0
            for i in range(len(mod._s.factor_blocks)):
                block = mod._s.factor_blocks[i]
                ix2 += block.k_factors
                T = self.filter_results.transition
                lag_names = []
                for j in range(block.factor_order):
                    lag_names += [f'L{j + 1}.{name}' for name in block.factor_names]
                data = pd.DataFrame(T[block.factors_L1, block.factors_ar, 0], index=block.factor_names, columns=lag_names)
                data.index.name = ''
                try:
                    data = data.map(lambda s: '%.2f' % s)
                except AttributeError:
                    data = data.applymap(lambda s: '%.2f' % s)
                Q = self.filter_results.state_cov
                if block.k_factors == 1:
                    data['   error variance'] = Q[ix1, ix1]
                else:
                    data['   error covariance'] = block.factor_names
                    for j in range(block.k_factors):
                        data[block.factor_names[j]] = Q[ix1:ix2, ix1 + j]
                cols_to_cast = data.columns[-block.k_factors:]
                data[cols_to_cast] = data[cols_to_cast].astype(object)
                try:
                    formatted_vals = data.iloc[:, -block.k_factors:].map(lambda s: f'{s:.2f}')
                except AttributeError:
                    formatted_vals = data.iloc[:, -block.k_factors:].applymap(lambda s: f'{s:.2f}')
                data.iloc[:, -block.k_factors:] = formatted_vals
                data = data.reset_index()
                params_data = data.values
                params_header = data.columns.tolist()
                params_stubs = None
                title = f'Transition: Factor block {i}'
                table = SimpleTable(params_data, params_header, params_stubs, txt_fmt=fmt_params, title=title)
                summary.tables.insert(table_ix, table)
                table_ix += 1
                ix1 = ix2
        return summary