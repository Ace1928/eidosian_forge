import warnings
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, bic, hqic
from statsmodels.tools.numdiff import approx_fprime_cs, approx_hess_cs
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tools.tools import Bunch, pinv_extended
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.regime_switching._hamilton_filter import (
from statsmodels.tsa.regime_switching._kim_smoother import (
from statsmodels.tsa.statespace.tools import (
class MarkovSwitchingResults(tsbase.TimeSeriesModelResults):
    """
    Class to hold results from fitting a Markov switching model

    Parameters
    ----------
    model : MarkovSwitching instance
        The fitted model instance
    params : ndarray
        Fitted parameters
    filter_results : HamiltonFilterResults or KimSmootherResults instance
        The underlying filter and, optionally, smoother output
    cov_type : str
        The type of covariance matrix estimator to use. Can be one of 'approx',
        'opg', 'robust', or 'none'.

    Attributes
    ----------
    model : Model instance
        A reference to the model that was fit.
    filter_results : HamiltonFilterResults or KimSmootherResults instance
        The underlying filter and, optionally, smoother output
    nobs : float
        The number of observations used to fit the model.
    params : ndarray
        The parameters of the model.
    scale : float
        This is currently set to 1.0 and not used by the model or its results.
    """
    use_t = False

    def __init__(self, model, params, results, cov_type='opg', cov_kwds=None, **kwargs):
        self.data = model.data
        tsbase.TimeSeriesModelResults.__init__(self, model, params, normalized_cov_params=None, scale=1.0)
        self.filter_results = results
        if isinstance(results, KimSmootherResults):
            self.smoother_results = results
        else:
            self.smoother_results = None
        self.nobs = model.nobs
        self.order = model.order
        self.k_regimes = model.k_regimes
        if not hasattr(self, 'cov_kwds'):
            self.cov_kwds = {}
        self.cov_type = cov_type
        self._cache = {}
        if cov_kwds is None:
            cov_kwds = {}
        self._cov_approx_complex_step = cov_kwds.pop('approx_complex_step', True)
        self._cov_approx_centered = cov_kwds.pop('approx_centered', False)
        try:
            self._rank = None
            self._get_robustcov_results(cov_type=cov_type, use_self=True, **cov_kwds)
        except np.linalg.LinAlgError:
            self._rank = 0
            k_params = len(self.params)
            self.cov_params_default = np.zeros((k_params, k_params)) * np.nan
            self.cov_kwds['cov_type'] = 'Covariance matrix could not be calculated: singular. information matrix.'
        attributes = ['regime_transition', 'initial_probabilities', 'conditional_loglikelihoods', 'predicted_marginal_probabilities', 'predicted_joint_probabilities', 'filtered_marginal_probabilities', 'filtered_joint_probabilities', 'joint_loglikelihoods', 'expected_durations']
        for name in attributes:
            setattr(self, name, getattr(self.filter_results, name))
        attributes = ['smoothed_joint_probabilities', 'smoothed_marginal_probabilities']
        for name in attributes:
            if self.smoother_results is not None:
                setattr(self, name, getattr(self.smoother_results, name))
            else:
                setattr(self, name, None)
        self.predicted_marginal_probabilities = self.predicted_marginal_probabilities.T
        self.filtered_marginal_probabilities = self.filtered_marginal_probabilities.T
        if self.smoother_results is not None:
            self.smoothed_marginal_probabilities = self.smoothed_marginal_probabilities.T
        if isinstance(self.data, PandasData):
            index = self.data.row_labels
            if self.expected_durations.ndim > 1:
                self.expected_durations = pd.DataFrame(self.expected_durations, index=index)
            self.predicted_marginal_probabilities = pd.DataFrame(self.predicted_marginal_probabilities, index=index)
            self.filtered_marginal_probabilities = pd.DataFrame(self.filtered_marginal_probabilities, index=index)
            if self.smoother_results is not None:
                self.smoothed_marginal_probabilities = pd.DataFrame(self.smoothed_marginal_probabilities, index=index)

    def _get_robustcov_results(self, cov_type='opg', **kwargs):
        from statsmodels.base.covtype import descriptions
        use_self = kwargs.pop('use_self', False)
        if use_self:
            res = self
        else:
            raise NotImplementedError
            res = self.__class__(self.model, self.params, normalized_cov_params=self.normalized_cov_params, scale=self.scale)
        res.cov_type = cov_type
        res.cov_kwds = {}
        approx_type_str = 'complex-step'
        k_params = len(self.params)
        if k_params == 0:
            res.cov_params_default = np.zeros((0, 0))
            res._rank = 0
            res.cov_kwds['description'] = 'No parameters estimated.'
        elif cov_type == 'custom':
            res.cov_type = kwargs['custom_cov_type']
            res.cov_params_default = kwargs['custom_cov_params']
            res.cov_kwds['description'] = kwargs['custom_description']
            res._rank = np.linalg.matrix_rank(res.cov_params_default)
        elif cov_type == 'none':
            res.cov_params_default = np.zeros((k_params, k_params)) * np.nan
            res._rank = np.nan
            res.cov_kwds['description'] = descriptions['none']
        elif self.cov_type == 'approx':
            res.cov_params_default = res.cov_params_approx
            res.cov_kwds['description'] = descriptions['approx'].format(approx_type=approx_type_str)
        elif self.cov_type == 'opg':
            res.cov_params_default = res.cov_params_opg
            res.cov_kwds['description'] = descriptions['OPG'].format(approx_type=approx_type_str)
        elif self.cov_type == 'robust':
            res.cov_params_default = res.cov_params_robust
            res.cov_kwds['description'] = descriptions['robust'].format(approx_type=approx_type_str)
        else:
            raise NotImplementedError('Invalid covariance matrix type.')
        return res

    @cache_readonly
    def aic(self):
        """
        (float) Akaike Information Criterion
        """
        return aic(self.llf, self.nobs, self.params.shape[0])

    @cache_readonly
    def bic(self):
        """
        (float) Bayes Information Criterion
        """
        return bic(self.llf, self.nobs, self.params.shape[0])

    @cache_readonly
    def cov_params_approx(self):
        """
        (array) The variance / covariance matrix. Computed using the numerical
        Hessian approximated by complex step or finite differences methods.
        """
        evaluated_hessian = self.model.hessian(self.params, transformed=True)
        neg_cov, singular_values = pinv_extended(evaluated_hessian)
        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))
        return -neg_cov

    @cache_readonly
    def cov_params_opg(self):
        """
        (array) The variance / covariance matrix. Computed using the outer
        product of gradients method.
        """
        score_obs = self.model.score_obs(self.params, transformed=True).T
        cov_params, singular_values = pinv_extended(np.inner(score_obs, score_obs))
        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))
        return cov_params

    @cache_readonly
    def cov_params_robust(self):
        """
        (array) The QMLE variance / covariance matrix. Computed using the
        numerical Hessian as the evaluated hessian.
        """
        cov_opg = self.cov_params_opg
        evaluated_hessian = self.model.hessian(self.params, transformed=True)
        cov_params, singular_values = pinv_extended(np.dot(np.dot(evaluated_hessian, cov_opg), evaluated_hessian))
        if self._rank is None:
            self._rank = np.linalg.matrix_rank(np.diag(singular_values))
        return cov_params

    @cache_readonly
    def fittedvalues(self):
        """
        (array) The predicted values of the model. An (nobs x k_endog) array.
        """
        return self.model.predict(self.params)

    @cache_readonly
    def hqic(self):
        """
        (float) Hannan-Quinn Information Criterion
        """
        return hqic(self.llf, self.nobs, self.params.shape[0])

    @cache_readonly
    def llf_obs(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        return self.model.loglikeobs(self.params)

    @cache_readonly
    def llf(self):
        """
        (float) The value of the log-likelihood function evaluated at `params`.
        """
        return self.model.loglike(self.params)

    @cache_readonly
    def resid(self):
        """
        (array) The model residuals. An (nobs x k_endog) array.
        """
        return self.model.endog - self.fittedvalues

    @property
    def joint_likelihoods(self):
        return np.exp(self.joint_loglikelihoods)

    def predict(self, start=None, end=None, probabilities=None, conditional=False):
        """
        In-sample prediction and out-of-sample forecasting

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
        probabilities : str or array_like, optional
            Specifies the weighting probabilities used in constructing the
            prediction as a weighted average. If a string, can be 'predicted',
            'filtered', or 'smoothed'. Otherwise can be an array of
            probabilities to use. Default is smoothed.
        conditional : bool or int, optional
            Whether or not to return predictions conditional on current or
            past regimes. If False, returns a single vector of weighted
            predictions. If True or 1, returns predictions conditional on the
            current regime. For larger integers, returns predictions
            conditional on the current regime and some number of past regimes.

        Returns
        -------
        predict : ndarray
            Array of out of in-sample predictions and / or out-of-sample
            forecasts. An (npredict x k_endog) array.
        """
        return self.model.predict(self.params, start=start, end=end, probabilities=probabilities, conditional=conditional)

    def forecast(self, steps=1, **kwargs):
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : int, str, or datetime, optional
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency, steps
            must be an integer. Default
        **kwargs
            Additional arguments may required for forecasting beyond the end
            of the sample. See `FilterResults.predict` for more details.

        Returns
        -------
        forecast : ndarray
            Array of out of sample forecasts. A (steps x k_endog) array.
        """
        raise NotImplementedError

    def summary(self, alpha=0.05, start=None, title=None, model_name=None, display_params=True):
        """
        Summarize the Model

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals. Default is 0.05.
        start : int, optional
            Integer of the start observation. Default is 0.
        title : str, optional
            The title of the summary table.
        model_name : str
            The name of the model used. Default is to use model class name.
        display_params : bool, optional
            Whether or not to display tables of estimated parameters. Default
            is True. Usually only used internally.

        Returns
        -------
        summary : Summary instance
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        from statsmodels.iolib.summary import Summary
        model = self.model
        if title is None:
            title = 'Markov Switching Model Results'
        if start is None:
            start = 0
        if self.data.dates is not None:
            dates = self.data.dates
            d = dates[start]
            sample = ['%02d-%02d-%02d' % (d.month, d.day, d.year)]
            d = dates[-1]
            sample += ['- ' + '%02d-%02d-%02d' % (d.month, d.day, d.year)]
        else:
            sample = [str(start), ' - ' + str(self.model.nobs)]
        if model_name is None:
            model_name = model.__class__.__name__
        if not isinstance(model_name, list):
            model_name = [model_name]
        top_left = [('Dep. Variable:', None)]
        top_left.append(('Model:', [model_name[0]]))
        for i in range(1, len(model_name)):
            top_left.append(('', ['+ ' + model_name[i]]))
        top_left += [('Date:', None), ('Time:', None), ('Sample:', [sample[0]]), ('', [sample[1]])]
        top_right = [('No. Observations:', [self.model.nobs]), ('Log Likelihood', ['%#5.3f' % self.llf]), ('AIC', ['%#5.3f' % self.aic]), ('BIC', ['%#5.3f' % self.bic]), ('HQIC', ['%#5.3f' % self.hqic])]
        if hasattr(self, 'cov_type'):
            top_left.append(('Covariance Type:', [self.cov_type]))
        summary = Summary()
        summary.add_table_2cols(self, gleft=top_left, gright=top_right, title=title)
        import re
        from statsmodels.iolib.summary import summary_params

        def make_table(self, mask, title, strip_end=True):
            res = (self, self.params[mask], self.bse[mask], self.tvalues[mask], self.pvalues[mask], self.conf_int(alpha)[mask])
            param_names = [re.sub('\\[\\d+\\]$', '', name) for name in np.array(self.data.param_names)[mask].tolist()]
            return summary_params(res, yname=None, xname=param_names, alpha=alpha, use_t=False, title=title)
        params = model.parameters
        regime_masks = [[] for i in range(model.k_regimes)]
        other_masks = {}
        for key, switching in params.switching.items():
            k_params = len(switching)
            if key == 'regime_transition':
                continue
            other_masks[key] = []
            for i in range(k_params):
                if switching[i]:
                    for j in range(self.k_regimes):
                        regime_masks[j].append(params[j, key][i])
                else:
                    other_masks[key].append(params[0, key][i])
        for i in range(self.k_regimes):
            mask = regime_masks[i]
            if len(mask) > 0:
                table = make_table(self, mask, 'Regime %d parameters' % i)
                summary.tables.append(table)
        mask = []
        for key, _mask in other_masks.items():
            mask.extend(_mask)
        if len(mask) > 0:
            table = make_table(self, mask, 'Non-switching parameters')
            summary.tables.append(table)
        mask = params['regime_transition']
        table = make_table(self, mask, 'Regime transition parameters')
        summary.tables.append(table)
        etext = []
        if hasattr(self, 'cov_type') and 'description' in self.cov_kwds:
            etext.append(self.cov_kwds['description'])
        if self._rank < len(self.params):
            etext.append('Covariance matrix is singular or near-singular, with condition number %6.3g. Standard errors may be unstable.' % _safe_cond(self.cov_params()))
        if etext:
            etext = [f'[{i + 1}] {text}' for i, text in enumerate(etext)]
            etext.insert(0, 'Warnings:')
            summary.add_extra_txt(etext)
        return summary