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
def _init_latent_vars(self, n_features, dtype=np.float64):
    """Initialize latent variables."""
    self.random_state_ = check_random_state(self.random_state)
    self.n_batch_iter_ = 1
    self.n_iter_ = 0
    if self.doc_topic_prior is None:
        self.doc_topic_prior_ = 1.0 / self.n_components
    else:
        self.doc_topic_prior_ = self.doc_topic_prior
    if self.topic_word_prior is None:
        self.topic_word_prior_ = 1.0 / self.n_components
    else:
        self.topic_word_prior_ = self.topic_word_prior
    init_gamma = 100.0
    init_var = 1.0 / init_gamma
    self.components_ = self.random_state_.gamma(init_gamma, init_var, (self.n_components, n_features)).astype(dtype, copy=False)
    self.exp_dirichlet_component_ = np.exp(_dirichlet_expectation_2d(self.components_))