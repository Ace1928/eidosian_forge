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
def _update_doc_distribution(X, exp_topic_word_distr, doc_topic_prior, max_doc_update_iter, mean_change_tol, cal_sstats, random_state):
    """E-step: update document-topic distribution.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Document word matrix.

    exp_topic_word_distr : ndarray of shape (n_topics, n_features)
        Exponential value of expectation of log topic word distribution.
        In the literature, this is `exp(E[log(beta)])`.

    doc_topic_prior : float
        Prior of document topic distribution `theta`.

    max_doc_update_iter : int
        Max number of iterations for updating document topic distribution in
        the E-step.

    mean_change_tol : float
        Stopping tolerance for updating document topic distribution in E-step.

    cal_sstats : bool
        Parameter that indicate to calculate sufficient statistics or not.
        Set `cal_sstats` to `True` when we need to run M-step.

    random_state : RandomState instance or None
        Parameter that indicate how to initialize document topic distribution.
        Set `random_state` to None will initialize document topic distribution
        to a constant number.

    Returns
    -------
    (doc_topic_distr, suff_stats) :
        `doc_topic_distr` is unnormalized topic distribution for each document.
        In the literature, this is `gamma`. we can calculate `E[log(theta)]`
        from it.
        `suff_stats` is expected sufficient statistics for the M-step.
            When `cal_sstats == False`, this will be None.

    """
    is_sparse_x = sp.issparse(X)
    n_samples, n_features = X.shape
    n_topics = exp_topic_word_distr.shape[0]
    if random_state:
        doc_topic_distr = random_state.gamma(100.0, 0.01, (n_samples, n_topics)).astype(X.dtype, copy=False)
    else:
        doc_topic_distr = np.ones((n_samples, n_topics), dtype=X.dtype)
    exp_doc_topic = np.exp(_dirichlet_expectation_2d(doc_topic_distr))
    suff_stats = np.zeros(exp_topic_word_distr.shape, dtype=X.dtype) if cal_sstats else None
    if is_sparse_x:
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr
    ctype = 'float' if X.dtype == np.float32 else 'double'
    mean_change = cy_mean_change[ctype]
    dirichlet_expectation_1d = cy_dirichlet_expectation_1d[ctype]
    eps = np.finfo(X.dtype).eps
    for idx_d in range(n_samples):
        if is_sparse_x:
            ids = X_indices[X_indptr[idx_d]:X_indptr[idx_d + 1]]
            cnts = X_data[X_indptr[idx_d]:X_indptr[idx_d + 1]]
        else:
            ids = np.nonzero(X[idx_d, :])[0]
            cnts = X[idx_d, ids]
        doc_topic_d = doc_topic_distr[idx_d, :]
        exp_doc_topic_d = exp_doc_topic[idx_d, :].copy()
        exp_topic_word_d = exp_topic_word_distr[:, ids]
        for _ in range(0, max_doc_update_iter):
            last_d = doc_topic_d
            norm_phi = np.dot(exp_doc_topic_d, exp_topic_word_d) + eps
            doc_topic_d = exp_doc_topic_d * np.dot(cnts / norm_phi, exp_topic_word_d.T)
            dirichlet_expectation_1d(doc_topic_d, doc_topic_prior, exp_doc_topic_d)
            if mean_change(last_d, doc_topic_d) < mean_change_tol:
                break
        doc_topic_distr[idx_d, :] = doc_topic_d
        if cal_sstats:
            norm_phi = np.dot(exp_doc_topic_d, exp_topic_word_d) + eps
            suff_stats[:, ids] += np.outer(exp_doc_topic_d, cnts / norm_phi)
    return (doc_topic_distr, suff_stats)