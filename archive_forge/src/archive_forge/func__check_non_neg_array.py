from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from joblib import effective_n_jobs
from scipy.special import gammaln, logsumexp
from ..base import (
from ..utils import check_random_state, gen_batches, gen_even_slices
from ..utils._param_validation import Interval, StrOptions
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted, check_non_negative
from ._online_lda_fast import (
from ._online_lda_fast import (
from ._online_lda_fast import (
def _check_non_neg_array(self, X, reset_n_features, whom):
    """check X format

        check X format and make sure no negative value in X.

        Parameters
        ----------
        X :  array-like or sparse matrix

        """
    dtype = [np.float64, np.float32] if reset_n_features else self.components_.dtype
    X = self._validate_data(X, reset=reset_n_features, accept_sparse='csr', dtype=dtype)
    check_non_negative(X, whom)
    return X