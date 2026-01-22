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
def _unnormalized_transform(self, X):
    """Transform data X according to fitted model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        Returns
        -------
        doc_topic_distr : ndarray of shape (n_samples, n_components)
            Document topic distribution for X.
        """
    doc_topic_distr, _ = self._e_step(X, cal_sstats=False, random_init=False)
    return doc_topic_distr