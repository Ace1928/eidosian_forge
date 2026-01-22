from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import itertools
import numpy as np
from statsmodels.gam.smooth_basis import (GenericSmoothers,
class MultivariateGAMCV(BaseCV):

    def __init__(self, smoother, alphas, gam, cost, endog, exog, cv_iterator):
        self.cost = cost
        self.gam = gam
        self.smoother = smoother
        self.exog_linear = exog
        self.alphas = alphas
        self.cv_iterator = cv_iterator
        super().__init__(cv_iterator, endog, self.smoother.basis)

    def _error(self, train_index, test_index, **kwargs):
        train_smoother, test_smoother = _split_train_test_smoothers(self.smoother.x, self.smoother, train_index, test_index)
        endog_train = self.endog[train_index]
        endog_test = self.endog[test_index]
        if self.exog_linear is not None:
            exog_linear_train = self.exog_linear[train_index]
            exog_linear_test = self.exog_linear[test_index]
        else:
            exog_linear_train = None
            exog_linear_test = None
        gam = self.gam(endog_train, exog=exog_linear_train, smoother=train_smoother, alpha=self.alphas)
        gam_res = gam.fit(**kwargs)
        endog_est = gam_res.predict(exog_linear_test, test_smoother.basis, transform=False)
        return self.cost(endog_test, endog_est)