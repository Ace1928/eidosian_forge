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