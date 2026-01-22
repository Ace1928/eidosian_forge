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
class MarkovSwitching(tsbase.TimeSeriesModel):
    """
    First-order k-regime Markov switching model

    Parameters
    ----------
    endog : array_like
        The endogenous variable.
    k_regimes : int
        The number of regimes.
    order : int, optional
        The order of the model describes the dependence of the likelihood on
        previous regimes. This depends on the model in question and should be
        set appropriately by subclasses.
    exog_tvtp : array_like, optional
        Array of exogenous or lagged variables to use in calculating
        time-varying transition probabilities (TVTP). TVTP is only used if this
        variable is provided. If an intercept is desired, a column of ones must
        be explicitly included in this array.

    Notes
    -----
    This model is new and API stability is not guaranteed, although changes
    will be made in a backwards compatible way if possible.

    References
    ----------
    Kim, Chang-Jin, and Charles R. Nelson. 1999.
    "State-Space Models with Regime Switching:
    Classical and Gibbs-Sampling Approaches with Applications".
    MIT Press Books. The MIT Press.
    """

    def __init__(self, endog, k_regimes, order=0, exog_tvtp=None, exog=None, dates=None, freq=None, missing='none'):
        self.k_regimes = k_regimes
        self.tvtp = exog_tvtp is not None
        self.order = order
        self.k_tvtp, self.exog_tvtp = prepare_exog(exog_tvtp)
        super().__init__(endog, exog, dates=dates, freq=freq, missing=missing)
        self.nobs = self.endog.shape[0]
        if self.endog.ndim > 1 and self.endog.shape[1] > 1:
            raise ValueError('Must have univariate endogenous data.')
        if self.k_regimes < 2:
            raise ValueError('Markov switching models must have at least two regimes.')
        if not (self.exog_tvtp is None or self.exog_tvtp.shape[0] == self.nobs):
            raise ValueError('Time-varying transition probabilities exogenous array must have the same number of observations as the endogenous array.')
        self.parameters = MarkovSwitchingParams(self.k_regimes)
        k_transition = self.k_regimes - 1
        if self.tvtp:
            k_transition *= self.k_tvtp
        self.parameters['regime_transition'] = [1] * k_transition
        self._initialization = 'steady-state'
        self._initial_probabilities = None

    @property
    def k_params(self):
        """
        (int) Number of parameters in the model
        """
        return self.parameters.k_params

    def initialize_steady_state(self):
        """
        Set initialization of regime probabilities to be steady-state values

        Notes
        -----
        Only valid if there are not time-varying transition probabilities.
        """
        if self.tvtp:
            raise ValueError('Cannot use steady-state initialization when the regime transition matrix is time-varying.')
        self._initialization = 'steady-state'
        self._initial_probabilities = None

    def initialize_known(self, probabilities, tol=1e-08):
        """
        Set initialization of regime probabilities to use known values
        """
        self._initialization = 'known'
        probabilities = np.array(probabilities, ndmin=1)
        if not probabilities.shape == (self.k_regimes,):
            raise ValueError('Initial probabilities must be a vector of shape (k_regimes,).')
        if not np.abs(np.sum(probabilities) - 1) < tol:
            raise ValueError('Initial probabilities vector must sum to one.')
        self._initial_probabilities = probabilities

    def initial_probabilities(self, params, regime_transition=None):
        """
        Retrieve initial probabilities
        """
        params = np.array(params, ndmin=1)
        if self._initialization == 'steady-state':
            if regime_transition is None:
                regime_transition = self.regime_transition_matrix(params)
            if regime_transition.ndim == 3:
                regime_transition = regime_transition[..., 0]
            m = regime_transition.shape[0]
            A = np.c_[(np.eye(m) - regime_transition).T, np.ones(m)].T
            try:
                probabilities = np.linalg.pinv(A)[:, -1]
            except np.linalg.LinAlgError:
                raise RuntimeError('Steady-state probabilities could not be constructed.')
        elif self._initialization == 'known':
            probabilities = self._initial_probabilities
        else:
            raise RuntimeError('Invalid initialization method selected.')
        probabilities = np.maximum(probabilities, 1e-20)
        return probabilities

    def _regime_transition_matrix_tvtp(self, params, exog_tvtp=None):
        if exog_tvtp is None:
            exog_tvtp = self.exog_tvtp
        nobs = len(exog_tvtp)
        regime_transition_matrix = np.zeros((self.k_regimes, self.k_regimes, nobs), dtype=np.promote_types(np.float64, params.dtype))
        for i in range(self.k_regimes):
            coeffs = params[self.parameters[i, 'regime_transition']]
            regime_transition_matrix[:-1, i, :] = np.dot(exog_tvtp, np.reshape(coeffs, (self.k_regimes - 1, self.k_tvtp)).T).T
        tmp = np.c_[np.zeros((nobs, self.k_regimes, 1)), regime_transition_matrix[:-1, :, :].T].T
        regime_transition_matrix[:-1, :, :] = np.exp(regime_transition_matrix[:-1, :, :] - logsumexp(tmp, axis=0))
        regime_transition_matrix[-1, :, :] = 1 - np.sum(regime_transition_matrix[:-1, :, :], axis=0)
        return regime_transition_matrix

    def regime_transition_matrix(self, params, exog_tvtp=None):
        """
        Construct the left-stochastic transition matrix

        Notes
        -----
        This matrix will either be shaped (k_regimes, k_regimes, 1) or if there
        are time-varying transition probabilities, it will be shaped
        (k_regimes, k_regimes, nobs).

        The (i,j)th element of this matrix is the probability of transitioning
        from regime j to regime i; thus the previous regime is represented in a
        column and the next regime is represented by a row.

        It is left-stochastic, meaning that each column sums to one (because
        it is certain that from one regime (j) you will transition to *some
        other regime*).
        """
        params = np.array(params, ndmin=1)
        if not self.tvtp:
            regime_transition_matrix = np.zeros((self.k_regimes, self.k_regimes, 1), dtype=np.promote_types(np.float64, params.dtype))
            regime_transition_matrix[:-1, :, 0] = np.reshape(params[self.parameters['regime_transition']], (self.k_regimes - 1, self.k_regimes))
            regime_transition_matrix[-1, :, 0] = 1 - np.sum(regime_transition_matrix[:-1, :, 0], axis=0)
        else:
            regime_transition_matrix = self._regime_transition_matrix_tvtp(params, exog_tvtp)
        return regime_transition_matrix

    def predict(self, params, start=None, end=None, probabilities=None, conditional=False):
        """
        In-sample prediction and out-of-sample forecasting

        Parameters
        ----------
        params : ndarray
            Parameters at which to form predictions
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
            forecasts.
        """
        if start is None:
            start = self._index[0]
        start, end, out_of_sample, prediction_index = self._get_prediction_index(start, end)
        if out_of_sample > 0:
            raise NotImplementedError
        predict = self.predict_conditional(params)
        squeezed = np.squeeze(predict)
        if squeezed.ndim - 1 > conditional:
            if probabilities is None or probabilities == 'smoothed':
                results = self.smooth(params, return_raw=True)
                probabilities = results.smoothed_joint_probabilities
            elif probabilities == 'filtered':
                results = self.filter(params, return_raw=True)
                probabilities = results.filtered_joint_probabilities
            elif probabilities == 'predicted':
                results = self.filter(params, return_raw=True)
                probabilities = results.predicted_joint_probabilities
            predict = predict * probabilities
            for i in range(predict.ndim - 1 - int(conditional)):
                predict = np.sum(predict, axis=-2)
        else:
            predict = squeezed
        return predict[start:end + out_of_sample + 1]

    def predict_conditional(self, params):
        """
        In-sample prediction, conditional on the current, and possibly past,
        regimes

        Parameters
        ----------
        params : array_like
            Array of parameters at which to perform prediction.

        Returns
        -------
        predict : array_like
            Array of predictions conditional on current, and possibly past,
            regimes
        """
        raise NotImplementedError

    def _conditional_loglikelihoods(self, params):
        """
        Compute likelihoods conditional on the current period's regime (and
        the last self.order periods' regimes if self.order > 0).

        Must be implemented in subclasses.
        """
        raise NotImplementedError

    def _filter(self, params, regime_transition=None):
        if regime_transition is None:
            regime_transition = self.regime_transition_matrix(params)
        initial_probabilities = self.initial_probabilities(params, regime_transition)
        conditional_loglikelihoods = self._conditional_loglikelihoods(params)
        return (regime_transition, initial_probabilities, conditional_loglikelihoods) + cy_hamilton_filter_log(initial_probabilities, regime_transition, conditional_loglikelihoods, self.order)

    def filter(self, params, transformed=True, cov_type=None, cov_kwds=None, return_raw=False, results_class=None, results_wrapper_class=None):
        """
        Apply the Hamilton filter

        Parameters
        ----------
        params : array_like
            Array of parameters at which to perform filtering.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        cov_type : str, optional
            See `fit` for a description of covariance matrix types
            for results object.
        cov_kwds : dict or None, optional
            See `fit` for a description of required keywords for alternative
            covariance estimators
        return_raw : bool,optional
            Whether or not to return only the raw Hamilton filter output or a
            full results object. Default is to return a full results object.
        results_class : type, optional
            A results class to instantiate rather than
            `MarkovSwitchingResults`. Usually only used internally by
            subclasses.
        results_wrapper_class : type, optional
            A results wrapper class to instantiate rather than
            `MarkovSwitchingResults`. Usually only used internally by
            subclasses.

        Returns
        -------
        MarkovSwitchingResults
        """
        params = np.array(params, ndmin=1)
        if not transformed:
            params = self.transform_params(params)
        self.data.param_names = self.param_names
        names = ['regime_transition', 'initial_probabilities', 'conditional_loglikelihoods', 'filtered_marginal_probabilities', 'predicted_joint_probabilities', 'joint_loglikelihoods', 'filtered_joint_probabilities', 'predicted_joint_probabilities_log', 'filtered_joint_probabilities_log']
        result = HamiltonFilterResults(self, Bunch(**dict(zip(names, self._filter(params)))))
        return self._wrap_results(params, result, return_raw, cov_type, cov_kwds, results_class, results_wrapper_class)

    def _smooth(self, params, predicted_joint_probabilities_log, filtered_joint_probabilities_log, regime_transition=None):
        if regime_transition is None:
            regime_transition = self.regime_transition_matrix(params)
        return cy_kim_smoother_log(regime_transition, predicted_joint_probabilities_log, filtered_joint_probabilities_log)

    @property
    def _res_classes(self):
        return {'fit': (MarkovSwitchingResults, MarkovSwitchingResultsWrapper)}

    def _wrap_results(self, params, result, return_raw, cov_type=None, cov_kwds=None, results_class=None, wrapper_class=None):
        if not return_raw:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            if cov_kwds is not None:
                result_kwargs['cov_kwds'] = cov_kwds
            if results_class is None:
                results_class = self._res_classes['fit'][0]
            if wrapper_class is None:
                wrapper_class = self._res_classes['fit'][1]
            res = results_class(self, params, result, **result_kwargs)
            result = wrapper_class(res)
        return result

    def smooth(self, params, transformed=True, cov_type=None, cov_kwds=None, return_raw=False, results_class=None, results_wrapper_class=None):
        """
        Apply the Kim smoother and Hamilton filter

        Parameters
        ----------
        params : array_like
            Array of parameters at which to perform filtering.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        cov_type : str, optional
            See `fit` for a description of covariance matrix types
            for results object.
        cov_kwds : dict or None, optional
            See `fit` for a description of required keywords for alternative
            covariance estimators
        return_raw : bool,optional
            Whether or not to return only the raw Hamilton filter output or a
            full results object. Default is to return a full results object.
        results_class : type, optional
            A results class to instantiate rather than
            `MarkovSwitchingResults`. Usually only used internally by
            subclasses.
        results_wrapper_class : type, optional
            A results wrapper class to instantiate rather than
            `MarkovSwitchingResults`. Usually only used internally by
            subclasses.

        Returns
        -------
        MarkovSwitchingResults
        """
        params = np.array(params, ndmin=1)
        if not transformed:
            params = self.transform_params(params)
        self.data.param_names = self.param_names
        names = ['regime_transition', 'initial_probabilities', 'conditional_loglikelihoods', 'filtered_marginal_probabilities', 'predicted_joint_probabilities', 'joint_loglikelihoods', 'filtered_joint_probabilities', 'predicted_joint_probabilities_log', 'filtered_joint_probabilities_log']
        result = Bunch(**dict(zip(names, self._filter(params))))
        out = self._smooth(params, result.predicted_joint_probabilities_log, result.filtered_joint_probabilities_log)
        result['smoothed_joint_probabilities'] = out[0]
        result['smoothed_marginal_probabilities'] = out[1]
        result = KimSmootherResults(self, result)
        return self._wrap_results(params, result, return_raw, cov_type, cov_kwds, results_class, results_wrapper_class)

    def loglikeobs(self, params, transformed=True):
        """
        Loglikelihood evaluation for each period

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        """
        params = np.array(params, ndmin=1)
        if not transformed:
            params = self.transform_params(params)
        results = self._filter(params)
        return results[5]

    def loglike(self, params, transformed=True):
        """
        Loglikelihood evaluation

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        """
        return np.sum(self.loglikeobs(params, transformed))

    def score(self, params, transformed=True):
        """
        Compute the score function at params.

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the score
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        """
        params = np.array(params, ndmin=1)
        return approx_fprime_cs(params, self.loglike, args=(transformed,))

    def score_obs(self, params, transformed=True):
        """
        Compute the score per observation, evaluated at params

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the score
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        """
        params = np.array(params, ndmin=1)
        return approx_fprime_cs(params, self.loglikeobs, args=(transformed,))

    def hessian(self, params, transformed=True):
        """
        Hessian matrix of the likelihood function, evaluated at the given
        parameters

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the Hessian
            function.
        transformed : bool, optional
            Whether or not `params` is already transformed. Default is True.
        """
        params = np.array(params, ndmin=1)
        return approx_hess_cs(params, self.loglike)

    def fit(self, start_params=None, transformed=True, cov_type='approx', cov_kwds=None, method='bfgs', maxiter=100, full_output=1, disp=0, callback=None, return_params=False, em_iter=5, search_reps=0, search_iter=5, search_scale=1.0, **kwargs):
        """
        Fits the model by maximum likelihood via Hamilton filter.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by Model.start_params.
        transformed : bool, optional
            Whether or not `start_params` is already transformed. Default is
            True.
        cov_type : str, optional
            The type of covariance matrix estimator to use. Can be one of
            'approx', 'opg', 'robust', or 'none'. Default is 'approx'.
        cov_kwds : dict or None, optional
            Keywords for alternative covariance estimators
        method : str, optional
            The `method` determines which solver from `scipy.optimize`
            is used, and it can be chosen from among the following strings:

            - 'newton' for Newton-Raphson, 'nm' for Nelder-Mead
            - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
            - 'lbfgs' for limited-memory BFGS with optional box constraints
            - 'powell' for modified Powell's method
            - 'cg' for conjugate gradient
            - 'ncg' for Newton-conjugate gradient
            - 'basinhopping' for global basin-hopping solver

            The explicit arguments in `fit` are passed to the solver,
            with the exception of the basin-hopping solver. Each
            solver has several optional arguments that are not the same across
            solvers. See the notes section below (or scipy.optimize) for the
            available arguments and for the list of explicit arguments that the
            basin-hopping solver supports.
        maxiter : int, optional
            The maximum number of iterations to perform.
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool, optional
            Set to True to print convergence messages.
        callback : callable callback(xk), optional
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        return_params : bool, optional
            Whether or not to return only the array of maximizing parameters.
            Default is False.
        em_iter : int, optional
            Number of initial EM iteration steps used to improve starting
            parameters.
        search_reps : int, optional
            Number of randomly drawn search parameters that are drawn around
            `start_params` to try and improve starting parameters. Default is
            0.
        search_iter : int, optional
            Number of initial EM iteration steps used to improve each of the
            search parameter repetitions.
        search_scale : float or array, optional.
            Scale of variates for random start parameter search.
        **kwargs
            Additional keyword arguments to pass to the optimizer.

        Returns
        -------
        MarkovSwitchingResults
        """
        if start_params is None:
            start_params = self.start_params
            transformed = True
        else:
            start_params = np.array(start_params, ndmin=1)
        if search_reps > 0:
            start_params = self._start_params_search(search_reps, start_params=start_params, transformed=transformed, em_iter=search_iter, scale=search_scale)
            transformed = True
        if em_iter and (not self.tvtp):
            start_params = self._fit_em(start_params, transformed=transformed, maxiter=em_iter, tolerance=0, return_params=True)
            transformed = True
        if transformed:
            start_params = self.untransform_params(start_params)
        fargs = (False,)
        mlefit = super().fit(start_params, method=method, fargs=fargs, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, skip_hessian=True, **kwargs)
        if return_params:
            result = self.transform_params(mlefit.params)
        else:
            result = self.smooth(mlefit.params, transformed=False, cov_type=cov_type, cov_kwds=cov_kwds)
            result.mlefit = mlefit
            result.mle_retvals = mlefit.mle_retvals
            result.mle_settings = mlefit.mle_settings
        return result

    def _fit_em(self, start_params=None, transformed=True, cov_type='none', cov_kwds=None, maxiter=50, tolerance=1e-06, full_output=True, return_params=False, **kwargs):
        """
        Fits the model using the Expectation-Maximization (EM) algorithm

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by `start_params`.
        transformed : bool, optional
            Whether or not `start_params` is already transformed. Default is
            True.
        cov_type : str, optional
            The type of covariance matrix estimator to use. Can be one of
            'approx', 'opg', 'robust', or 'none'. Default is 'none'.
        cov_kwds : dict or None, optional
            Keywords for alternative covariance estimators
        maxiter : int, optional
            The maximum number of iterations to perform.
        tolerance : float, optional
            The iteration stops when the difference between subsequent
            loglikelihood values is less than this tolerance.
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. This includes all intermediate values for
            parameters and loglikelihood values
        return_params : bool, optional
            Whether or not to return only the array of maximizing parameters.
            Default is False.
        **kwargs
            Additional keyword arguments to pass to the optimizer.

        Notes
        -----
        This is a private method for finding good starting parameters for MLE
        by scoring. It has not been tested for a thoroughly correct EM
        implementation in all cases. It does not support TVTP transition
        probabilities.

        Returns
        -------
        MarkovSwitchingResults
        """
        if start_params is None:
            start_params = self.start_params
            transformed = True
        else:
            start_params = np.array(start_params, ndmin=1)
        if not transformed:
            start_params = self.transform_params(start_params)
        llf = []
        params = [start_params]
        i = 0
        delta = 0
        while i < maxiter and (i < 2 or delta > tolerance):
            out = self._em_iteration(params[-1])
            llf.append(out[0].llf)
            params.append(out[1])
            if i > 0:
                delta = 2 * (llf[-1] - llf[-2]) / np.abs(llf[-1] + llf[-2])
            i += 1
        if return_params:
            result = params[-1]
        else:
            result = self.filter(params[-1], transformed=True, cov_type=cov_type, cov_kwds=cov_kwds)
            if full_output:
                em_retvals = Bunch(**{'params': np.array(params), 'llf': np.array(llf), 'iter': i})
                em_settings = Bunch(**{'tolerance': tolerance, 'maxiter': maxiter})
            else:
                em_retvals = None
                em_settings = None
            result.mle_retvals = em_retvals
            result.mle_settings = em_settings
        return result

    def _em_iteration(self, params0):
        """
        EM iteration

        Notes
        -----
        The EM iteration in this base class only performs the EM step for
        non-TVTP transition probabilities.
        """
        params1 = np.zeros(params0.shape, dtype=np.promote_types(np.float64, params0.dtype))
        result = self.smooth(params0, transformed=True, return_raw=True)
        if self.tvtp:
            params1[self.parameters['regime_transition']] = params0[self.parameters['regime_transition']]
        else:
            regime_transition = self._em_regime_transition(result)
            for i in range(self.k_regimes):
                params1[self.parameters[i, 'regime_transition']] = regime_transition[i]
        return (result, params1)

    def _em_regime_transition(self, result):
        """
        EM step for regime transition probabilities
        """
        tmp = result.smoothed_joint_probabilities
        for i in range(tmp.ndim - 3):
            tmp = np.sum(tmp, -2)
        smoothed_joint_probabilities = tmp
        k_transition = len(self.parameters[0, 'regime_transition'])
        regime_transition = np.zeros((self.k_regimes, k_transition))
        for i in range(self.k_regimes):
            for j in range(self.k_regimes - 1):
                regime_transition[i, j] = np.sum(smoothed_joint_probabilities[j, i]) / np.sum(result.smoothed_marginal_probabilities[i])
            delta = np.sum(regime_transition[i]) - 1
            if delta > 0:
                warnings.warn('Invalid regime transition probabilities estimated in EM iteration; probabilities have been re-scaled to continue estimation.', EstimationWarning)
                regime_transition[i] /= 1 + delta + 1e-06
        return regime_transition

    def _start_params_search(self, reps, start_params=None, transformed=True, em_iter=5, scale=1.0):
        """
        Search for starting parameters as random permutations of a vector

        Parameters
        ----------
        reps : int
            Number of random permutations to try.
        start_params : ndarray, optional
            Starting parameter vector. If not given, class-level start
            parameters are used.
        transformed : bool, optional
            If `start_params` was provided, whether or not those parameters
            are already transformed. Default is True.
        em_iter : int, optional
            Number of EM iterations to apply to each random permutation.
        scale : array or float, optional
            Scale of variates for random start parameter search. Can be given
            as an array of length equal to the number of parameters or as a
            single scalar.

        Notes
        -----
        This is a private method for finding good starting parameters for MLE
        by scoring, where the defaults have been set heuristically.
        """
        if start_params is None:
            start_params = self.start_params
            transformed = True
        else:
            start_params = np.array(start_params, ndmin=1)
        if transformed:
            start_params = self.untransform_params(start_params)
        scale = np.array(scale, ndmin=1)
        if scale.size == 1:
            scale = np.ones(self.k_params) * scale
        if not scale.size == self.k_params:
            raise ValueError('Scale of variates for random start parameter search must be given for each parameter or as a single scalar.')
        variates = np.zeros((reps, self.k_params))
        for i in range(self.k_params):
            variates[:, i] = scale[i] * np.random.uniform(-0.5, 0.5, size=reps)
        llf = self.loglike(start_params, transformed=False)
        params = start_params
        for i in range(reps):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    proposed_params = self._fit_em(start_params + variates[i], transformed=False, maxiter=em_iter, return_params=True)
                    proposed_llf = self.loglike(proposed_params)
                    if proposed_llf > llf:
                        llf = proposed_llf
                        params = self.untransform_params(proposed_params)
                except Exception:
                    pass
        return self.transform_params(params)

    @property
    def start_params(self):
        """
        (array) Starting parameters for maximum likelihood estimation.
        """
        params = np.zeros(self.k_params, dtype=np.float64)
        if self.tvtp:
            params[self.parameters['regime_transition']] = 0.0
        else:
            params[self.parameters['regime_transition']] = 1.0 / self.k_regimes
        return params

    @property
    def param_names(self):
        """
        (list of str) List of human readable parameter names (for parameters
        actually included in the model).
        """
        param_names = np.zeros(self.k_params, dtype=object)
        if self.tvtp:
            param_names[self.parameters['regime_transition']] = ['p[%d->%d].tvtp%d' % (j, i, k) for i in range(self.k_regimes - 1) for k in range(self.k_tvtp) for j in range(self.k_regimes)]
        else:
            param_names[self.parameters['regime_transition']] = ['p[%d->%d]' % (j, i) for i in range(self.k_regimes - 1) for j in range(self.k_regimes)]
        return param_names.tolist()

    def transform_params(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation

        Parameters
        ----------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer, to be
            transformed.

        Returns
        -------
        constrained : array_like
            Array of constrained parameters which may be used in likelihood
            evaluation.

        Notes
        -----
        In the base class, this only transforms the transition-probability-
        related parameters.
        """
        constrained = np.array(unconstrained, copy=True)
        constrained = constrained.astype(np.promote_types(np.float64, constrained.dtype))
        if self.tvtp:
            constrained[self.parameters['regime_transition']] = unconstrained[self.parameters['regime_transition']]
        else:
            for i in range(self.k_regimes):
                tmp1 = unconstrained[self.parameters[i, 'regime_transition']]
                tmp2 = np.r_[0, tmp1]
                constrained[self.parameters[i, 'regime_transition']] = np.exp(tmp1 - logsumexp(tmp2))
        return constrained

    def _untransform_logistic(self, unconstrained, constrained):
        """
        Function to allow using a numerical root-finder to reverse the
        logistic transform.
        """
        resid = np.zeros(unconstrained.shape, dtype=unconstrained.dtype)
        exp = np.exp(unconstrained)
        sum_exp = np.sum(exp)
        for i in range(len(unconstrained)):
            resid[i] = unconstrained[i] - np.log(1 + sum_exp - exp[i]) + np.log(1 / constrained[i] - 1)
        return resid

    def untransform_params(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer

        Parameters
        ----------
        constrained : array_like
            Array of constrained parameters used in likelihood evaluation, to
            be transformed.

        Returns
        -------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer.

        Notes
        -----
        In the base class, this only untransforms the transition-probability-
        related parameters.
        """
        unconstrained = np.array(constrained, copy=True)
        unconstrained = unconstrained.astype(np.promote_types(np.float64, unconstrained.dtype))
        if self.tvtp:
            unconstrained[self.parameters['regime_transition']] = constrained[self.parameters['regime_transition']]
        else:
            for i in range(self.k_regimes):
                s = self.parameters[i, 'regime_transition']
                if self.k_regimes == 2:
                    unconstrained[s] = -np.log(1.0 / constrained[s] - 1)
                else:
                    from scipy.optimize import root
                    out = root(self._untransform_logistic, np.zeros(unconstrained[s].shape, unconstrained.dtype), args=(constrained[s],))
                    if not out['success']:
                        raise ValueError('Could not untransform parameters.')
                    unconstrained[s] = out['x']
        return unconstrained