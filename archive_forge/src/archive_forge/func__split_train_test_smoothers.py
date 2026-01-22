from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import itertools
import numpy as np
from statsmodels.gam.smooth_basis import (GenericSmoothers,
def _split_train_test_smoothers(x, smoother, train_index, test_index):
    """split smoothers in test and train sets and create GenericSmoothers

    Note: this does not take exog_linear into account
    """
    train_smoothers = []
    test_smoothers = []
    for smoother in smoother.smoothers:
        train_basis = smoother.basis[train_index]
        train_der_basis = smoother.der_basis[train_index]
        train_der2_basis = smoother.der2_basis[train_index]
        train_cov_der2 = smoother.cov_der2
        train_x = smoother.x[train_index]
        train_smoothers.append(UnivariateGenericSmoother(train_x, train_basis, train_der_basis, train_der2_basis, train_cov_der2, smoother.variable_name + ' train'))
        test_basis = smoother.basis[test_index]
        test_der_basis = smoother.der_basis[test_index]
        test_cov_der2 = smoother.cov_der2
        test_x = smoother.x[test_index]
        test_smoothers.append(UnivariateGenericSmoother(test_x, test_basis, test_der_basis, train_der2_basis, test_cov_der2, smoother.variable_name + ' test'))
    train_multivariate_smoothers = GenericSmoothers(x[train_index], train_smoothers)
    test_multivariate_smoothers = GenericSmoothers(x[test_index], test_smoothers)
    return (train_multivariate_smoothers, test_multivariate_smoothers)