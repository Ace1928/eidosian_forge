import numbers
import numpy as np
from ..ensemble._bagging import _generate_indices
from ..metrics import check_scoring, get_scorer_names
from ..metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from ..model_selection._validation import _aggregate_score_dicts
from ..utils import Bunch, _safe_indexing, check_array, check_random_state
from ..utils._param_validation import (
from ..utils.parallel import Parallel, delayed
def _calculate_permutation_scores(estimator, X, y, sample_weight, col_idx, random_state, n_repeats, scorer, max_samples):
    """Calculate score when `col_idx` is permuted."""
    random_state = check_random_state(random_state)
    if max_samples < X.shape[0]:
        row_indices = _generate_indices(random_state=random_state, bootstrap=False, n_population=X.shape[0], n_samples=max_samples)
        X_permuted = _safe_indexing(X, row_indices, axis=0)
        y = _safe_indexing(y, row_indices, axis=0)
        if sample_weight is not None:
            sample_weight = _safe_indexing(sample_weight, row_indices, axis=0)
    else:
        X_permuted = X.copy()
    scores = []
    shuffling_idx = np.arange(X_permuted.shape[0])
    for _ in range(n_repeats):
        random_state.shuffle(shuffling_idx)
        if hasattr(X_permuted, 'iloc'):
            col = X_permuted.iloc[shuffling_idx, col_idx]
            col.index = X_permuted.index
            X_permuted[X_permuted.columns[col_idx]] = col
        else:
            X_permuted[:, col_idx] = X_permuted[shuffling_idx, col_idx]
        scores.append(_weights_scorer(scorer, estimator, X_permuted, y, sample_weight))
    if isinstance(scores[0], dict):
        scores = _aggregate_score_dicts(scores)
    else:
        scores = np.array(scores)
    return scores