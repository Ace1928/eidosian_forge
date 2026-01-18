from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import itertools
import numpy as np
from statsmodels.gam.smooth_basis import (GenericSmoothers,
k-fold cross-validation for GAM

    Warning: The API of this class is preliminary and will change.

    Parameters
    ----------
    smoother : additive smoother instance
    alphas : list of iteratables
        list of alpha for smooths. The product space will be used as alpha
        grid for cross-validation
    gam : model class
        model class for creating a model with k-fole training data
    cost : function
        cost function for the prediction error
    endog : ndarray
        dependent (response) variable of the model
    cv_iterator : instance of cross-validation iterator
    