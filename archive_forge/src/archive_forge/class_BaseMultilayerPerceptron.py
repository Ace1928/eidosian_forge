import warnings
from abc import ABCMeta, abstractmethod
from itertools import chain
from numbers import Integral, Real
import numpy as np
import scipy.optimize
from ..base import (
from ..exceptions import ConvergenceWarning
from ..metrics import accuracy_score, r2_score
from ..model_selection import train_test_split
from ..preprocessing import LabelBinarizer
from ..utils import (
from ..utils._param_validation import Interval, Options, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import (
from ..utils.optimize import _check_optimize_result
from ..utils.validation import check_is_fitted
from ._base import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS
from ._stochastic_optimizers import AdamOptimizer, SGDOptimizer
class BaseMultilayerPerceptron(BaseEstimator, metaclass=ABCMeta):
    """Base class for MLP classification and regression.

    Warning: This class should not be used directly.
    Use derived classes instead.

    .. versionadded:: 0.18
    """
    _parameter_constraints: dict = {'hidden_layer_sizes': ['array-like', Interval(Integral, 1, None, closed='left')], 'activation': [StrOptions({'identity', 'logistic', 'tanh', 'relu'})], 'solver': [StrOptions({'lbfgs', 'sgd', 'adam'})], 'alpha': [Interval(Real, 0, None, closed='left')], 'batch_size': [StrOptions({'auto'}), Interval(Integral, 1, None, closed='left')], 'learning_rate': [StrOptions({'constant', 'invscaling', 'adaptive'})], 'learning_rate_init': [Interval(Real, 0, None, closed='neither')], 'power_t': [Interval(Real, 0, None, closed='left')], 'max_iter': [Interval(Integral, 1, None, closed='left')], 'shuffle': ['boolean'], 'random_state': ['random_state'], 'tol': [Interval(Real, 0, None, closed='left')], 'verbose': ['verbose'], 'warm_start': ['boolean'], 'momentum': [Interval(Real, 0, 1, closed='both')], 'nesterovs_momentum': ['boolean'], 'early_stopping': ['boolean'], 'validation_fraction': [Interval(Real, 0, 1, closed='left')], 'beta_1': [Interval(Real, 0, 1, closed='left')], 'beta_2': [Interval(Real, 0, 1, closed='left')], 'epsilon': [Interval(Real, 0, None, closed='neither')], 'n_iter_no_change': [Interval(Integral, 1, None, closed='left'), Options(Real, {np.inf})], 'max_fun': [Interval(Integral, 1, None, closed='left')]}

    @abstractmethod
    def __init__(self, hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate, learning_rate_init, power_t, max_iter, loss, shuffle, random_state, tol, verbose, warm_start, momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, max_fun):
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.loss = loss
        self.hidden_layer_sizes = hidden_layer_sizes
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun

    def _unpack(self, packed_parameters):
        """Extract the coefficients and intercepts from packed_parameters."""
        for i in range(self.n_layers_ - 1):
            start, end, shape = self._coef_indptr[i]
            self.coefs_[i] = np.reshape(packed_parameters[start:end], shape)
            start, end = self._intercept_indptr[i]
            self.intercepts_[i] = packed_parameters[start:end]

    def _forward_pass(self, activations):
        """Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer.

        Parameters
        ----------
        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        """
        hidden_activation = ACTIVATIONS[self.activation]
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = safe_sparse_dot(activations[i], self.coefs_[i])
            activations[i + 1] += self.intercepts_[i]
            if i + 1 != self.n_layers_ - 1:
                hidden_activation(activations[i + 1])
        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activations[i + 1])
        return activations

    def _forward_pass_fast(self, X, check_input=True):
        """Predict using the trained model

        This is the same as _forward_pass but does not record the activations
        of all layers and only returns the last layer's activation.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        check_input : bool, default=True
            Perform input data validation or not.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The decision function of the samples for each class in the model.
        """
        if check_input:
            X = self._validate_data(X, accept_sparse=['csr', 'csc'], reset=False)
        activation = X
        hidden_activation = ACTIVATIONS[self.activation]
        for i in range(self.n_layers_ - 1):
            activation = safe_sparse_dot(activation, self.coefs_[i])
            activation += self.intercepts_[i]
            if i != self.n_layers_ - 2:
                hidden_activation(activation)
        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activation)
        return activation

    def _compute_loss_grad(self, layer, n_samples, activations, deltas, coef_grads, intercept_grads):
        """Compute the gradient of loss with respect to coefs and intercept for
        specified layer.

        This function does backpropagation for the specified one layer.
        """
        coef_grads[layer] = safe_sparse_dot(activations[layer].T, deltas[layer])
        coef_grads[layer] += self.alpha * self.coefs_[layer]
        coef_grads[layer] /= n_samples
        intercept_grads[layer] = np.mean(deltas[layer], 0)

    def _loss_grad_lbfgs(self, packed_coef_inter, X, y, activations, deltas, coef_grads, intercept_grads):
        """Compute the MLP loss function and its corresponding derivatives
        with respect to the different parameters given in the initialization.

        Returned gradients are packed in a single vector so it can be used
        in lbfgs

        Parameters
        ----------
        packed_coef_inter : ndarray
            A vector comprising the flattened coefficients and intercepts.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,)
            The target values.

        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.

        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function

        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.

        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.

        Returns
        -------
        loss : float
        grad : array-like, shape (number of nodes of all layers,)
        """
        self._unpack(packed_coef_inter)
        loss, coef_grads, intercept_grads = self._backprop(X, y, activations, deltas, coef_grads, intercept_grads)
        grad = _pack(coef_grads, intercept_grads)
        return (loss, grad)

    def _backprop(self, X, y, activations, deltas, coef_grads, intercept_grads):
        """Compute the MLP loss function and its corresponding derivatives
        with respect to each parameter: weights and bias vectors.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,)
            The target values.

        activations : list, length = n_layers - 1
             The ith element of the list holds the values of the ith layer.

        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function

        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.

        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.

        Returns
        -------
        loss : float
        coef_grads : list, length = n_layers - 1
        intercept_grads : list, length = n_layers - 1
        """
        n_samples = X.shape[0]
        activations = self._forward_pass(activations)
        loss_func_name = self.loss
        if loss_func_name == 'log_loss' and self.out_activation_ == 'logistic':
            loss_func_name = 'binary_log_loss'
        loss = LOSS_FUNCTIONS[loss_func_name](y, activations[-1])
        values = 0
        for s in self.coefs_:
            s = s.ravel()
            values += np.dot(s, s)
        loss += 0.5 * self.alpha * values / n_samples
        last = self.n_layers_ - 2
        deltas[last] = activations[-1] - y
        self._compute_loss_grad(last, n_samples, activations, deltas, coef_grads, intercept_grads)
        inplace_derivative = DERIVATIVES[self.activation]
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative(activations[i], deltas[i - 1])
            self._compute_loss_grad(i - 1, n_samples, activations, deltas, coef_grads, intercept_grads)
        return (loss, coef_grads, intercept_grads)

    def _initialize(self, y, layer_units, dtype):
        self.n_iter_ = 0
        self.t_ = 0
        self.n_outputs_ = y.shape[1]
        self.n_layers_ = len(layer_units)
        if not is_classifier(self):
            self.out_activation_ = 'identity'
        elif self._label_binarizer.y_type_ == 'multiclass':
            self.out_activation_ = 'softmax'
        else:
            self.out_activation_ = 'logistic'
        self.coefs_ = []
        self.intercepts_ = []
        for i in range(self.n_layers_ - 1):
            coef_init, intercept_init = self._init_coef(layer_units[i], layer_units[i + 1], dtype)
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)
        if self.solver in _STOCHASTIC_SOLVERS:
            self.loss_curve_ = []
            self._no_improvement_count = 0
            if self.early_stopping:
                self.validation_scores_ = []
                self.best_validation_score_ = -np.inf
                self.best_loss_ = None
            else:
                self.best_loss_ = np.inf
                self.validation_scores_ = None
                self.best_validation_score_ = None

    def _init_coef(self, fan_in, fan_out, dtype):
        factor = 6.0
        if self.activation == 'logistic':
            factor = 2.0
        init_bound = np.sqrt(factor / (fan_in + fan_out))
        coef_init = self._random_state.uniform(-init_bound, init_bound, (fan_in, fan_out))
        intercept_init = self._random_state.uniform(-init_bound, init_bound, fan_out)
        coef_init = coef_init.astype(dtype, copy=False)
        intercept_init = intercept_init.astype(dtype, copy=False)
        return (coef_init, intercept_init)

    def _fit(self, X, y, incremental=False):
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, '__iter__'):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)
        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError('hidden_layer_sizes must be > 0, got %s.' % hidden_layer_sizes)
        first_pass = not hasattr(self, 'coefs_') or (not self.warm_start and (not incremental))
        X, y = self._validate_input(X, y, incremental, reset=first_pass)
        n_samples, n_features = X.shape
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        self.n_outputs_ = y.shape[1]
        layer_units = [n_features] + hidden_layer_sizes + [self.n_outputs_]
        self._random_state = check_random_state(self.random_state)
        if first_pass:
            self._initialize(y, layer_units, X.dtype)
        activations = [X] + [None] * (len(layer_units) - 1)
        deltas = [None] * (len(activations) - 1)
        coef_grads = [np.empty((n_fan_in_, n_fan_out_), dtype=X.dtype) for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])]
        intercept_grads = [np.empty(n_fan_out_, dtype=X.dtype) for n_fan_out_ in layer_units[1:]]
        if self.solver in _STOCHASTIC_SOLVERS:
            self._fit_stochastic(X, y, activations, deltas, coef_grads, intercept_grads, layer_units, incremental)
        elif self.solver == 'lbfgs':
            self._fit_lbfgs(X, y, activations, deltas, coef_grads, intercept_grads, layer_units)
        weights = chain(self.coefs_, self.intercepts_)
        if not all((np.isfinite(w).all() for w in weights)):
            raise ValueError('Solver produced non-finite parameter weights. The input data may contain large values and need to be preprocessed.')
        return self

    def _fit_lbfgs(self, X, y, activations, deltas, coef_grads, intercept_grads, layer_units):
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = (layer_units[i], layer_units[i + 1])
            end = start + n_fan_in * n_fan_out
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end
        packed_coef_inter = _pack(self.coefs_, self.intercepts_)
        if self.verbose is True or self.verbose >= 1:
            iprint = 1
        else:
            iprint = -1
        opt_res = scipy.optimize.minimize(self._loss_grad_lbfgs, packed_coef_inter, method='L-BFGS-B', jac=True, options={'maxfun': self.max_fun, 'maxiter': self.max_iter, 'iprint': iprint, 'gtol': self.tol}, args=(X, y, activations, deltas, coef_grads, intercept_grads))
        self.n_iter_ = _check_optimize_result('lbfgs', opt_res, self.max_iter)
        self.loss_ = opt_res.fun
        self._unpack(opt_res.x)

    def _fit_stochastic(self, X, y, activations, deltas, coef_grads, intercept_grads, layer_units, incremental):
        params = self.coefs_ + self.intercepts_
        if not incremental or not hasattr(self, '_optimizer'):
            if self.solver == 'sgd':
                self._optimizer = SGDOptimizer(params, self.learning_rate_init, self.learning_rate, self.momentum, self.nesterovs_momentum, self.power_t)
            elif self.solver == 'adam':
                self._optimizer = AdamOptimizer(params, self.learning_rate_init, self.beta_1, self.beta_2, self.epsilon)
        if self.early_stopping and incremental:
            raise ValueError('partial_fit does not support early_stopping=True')
        early_stopping = self.early_stopping
        if early_stopping:
            should_stratify = is_classifier(self) and self.n_outputs_ == 1
            stratify = y if should_stratify else None
            X, X_val, y, y_val = train_test_split(X, y, random_state=self._random_state, test_size=self.validation_fraction, stratify=stratify)
            if is_classifier(self):
                y_val = self._label_binarizer.inverse_transform(y_val)
        else:
            X_val = None
            y_val = None
        n_samples = X.shape[0]
        sample_idx = np.arange(n_samples, dtype=int)
        if self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            if self.batch_size > n_samples:
                warnings.warn('Got `batch_size` less than 1 or larger than sample size. It is going to be clipped')
            batch_size = np.clip(self.batch_size, 1, n_samples)
        try:
            self.n_iter_ = 0
            for it in range(self.max_iter):
                if self.shuffle:
                    sample_idx = shuffle(sample_idx, random_state=self._random_state)
                accumulated_loss = 0.0
                for batch_slice in gen_batches(n_samples, batch_size):
                    if self.shuffle:
                        X_batch = _safe_indexing(X, sample_idx[batch_slice])
                        y_batch = y[sample_idx[batch_slice]]
                    else:
                        X_batch = X[batch_slice]
                        y_batch = y[batch_slice]
                    activations[0] = X_batch
                    batch_loss, coef_grads, intercept_grads = self._backprop(X_batch, y_batch, activations, deltas, coef_grads, intercept_grads)
                    accumulated_loss += batch_loss * (batch_slice.stop - batch_slice.start)
                    grads = coef_grads + intercept_grads
                    self._optimizer.update_params(params, grads)
                self.n_iter_ += 1
                self.loss_ = accumulated_loss / X.shape[0]
                self.t_ += n_samples
                self.loss_curve_.append(self.loss_)
                if self.verbose:
                    print('Iteration %d, loss = %.8f' % (self.n_iter_, self.loss_))
                self._update_no_improvement_count(early_stopping, X_val, y_val)
                self._optimizer.iteration_ends(self.t_)
                if self._no_improvement_count > self.n_iter_no_change:
                    if early_stopping:
                        msg = 'Validation score did not improve more than tol=%f for %d consecutive epochs.' % (self.tol, self.n_iter_no_change)
                    else:
                        msg = 'Training loss did not improve more than tol=%f for %d consecutive epochs.' % (self.tol, self.n_iter_no_change)
                    is_stopping = self._optimizer.trigger_stopping(msg, self.verbose)
                    if is_stopping:
                        break
                    else:
                        self._no_improvement_count = 0
                if incremental:
                    break
                if self.n_iter_ == self.max_iter:
                    warnings.warn("Stochastic Optimizer: Maximum iterations (%d) reached and the optimization hasn't converged yet." % self.max_iter, ConvergenceWarning)
        except KeyboardInterrupt:
            warnings.warn('Training interrupted by user.')
        if early_stopping:
            self.coefs_ = self._best_coefs
            self.intercepts_ = self._best_intercepts

    def _update_no_improvement_count(self, early_stopping, X_val, y_val):
        if early_stopping:
            self.validation_scores_.append(self._score(X_val, y_val))
            if self.verbose:
                print('Validation score: %f' % self.validation_scores_[-1])
            last_valid_score = self.validation_scores_[-1]
            if last_valid_score < self.best_validation_score_ + self.tol:
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0
            if last_valid_score > self.best_validation_score_:
                self.best_validation_score_ = last_valid_score
                self._best_coefs = [c.copy() for c in self.coefs_]
                self._best_intercepts = [i.copy() for i in self.intercepts_]
        else:
            if self.loss_curve_[-1] > self.best_loss_ - self.tol:
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0
            if self.loss_curve_[-1] < self.best_loss_:
                self.best_loss_ = self.loss_curve_[-1]

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns a trained MLP model.
        """
        return self._fit(X, y, incremental=False)

    def _check_solver(self):
        if self.solver not in _STOCHASTIC_SOLVERS:
            raise AttributeError('partial_fit is only available for stochastic optimizers. %s is not stochastic.' % self.solver)
        return True