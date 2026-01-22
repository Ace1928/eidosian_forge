import itertools
import sys
import time
from numbers import Integral, Real
from warnings import warn
import numpy as np
from joblib import effective_n_jobs
from scipy import linalg
from ..base import (
from ..linear_model import Lars, Lasso, LassoLars, orthogonal_mp_gram
from ..utils import check_array, check_random_state, gen_batches, gen_even_slices
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.extmath import randomized_svd, row_norms, svd_flip
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted
def _initialize_dict(self, X, random_state):
    """Initialization of the dictionary."""
    if self.dict_init is not None:
        dictionary = self.dict_init
    else:
        _, S, dictionary = randomized_svd(X, self._n_components, random_state=random_state)
        dictionary = S[:, np.newaxis] * dictionary
    if self._n_components <= len(dictionary):
        dictionary = dictionary[:self._n_components, :]
    else:
        dictionary = np.concatenate((dictionary, np.zeros((self._n_components - len(dictionary), dictionary.shape[1]), dtype=dictionary.dtype)))
    dictionary = check_array(dictionary, order='F', dtype=X.dtype, copy=False)
    dictionary = np.require(dictionary, requirements='W')
    return dictionary