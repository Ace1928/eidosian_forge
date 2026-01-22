from itertools import combinations
import numpy as np
from ..utils import check_array, check_consistent_length
from ..utils.multiclass import type_of_target
def _average_multiclass_ovo_score(binary_metric, y_true, y_score, average='macro'):
    """Average one-versus-one scores for multiclass classification.

    Uses the binary metric for one-vs-one multiclass classification,
    where the score is computed according to the Hand & Till (2001) algorithm.

    Parameters
    ----------
    binary_metric : callable
        The binary metric function to use that accepts the following as input:
            y_true_target : array, shape = [n_samples_target]
                Some sub-array of y_true for a pair of classes designated
                positive and negative in the one-vs-one scheme.
            y_score_target : array, shape = [n_samples_target]
                Scores corresponding to the probability estimates
                of a sample belonging to the designated positive class label

    y_true : array-like of shape (n_samples,)
        True multiclass labels.

    y_score : array-like of shape (n_samples, n_classes)
        Target scores corresponding to probability estimates of a sample
        belonging to a particular class.

    average : {'macro', 'weighted'}, default='macro'
        Determines the type of averaging performed on the pairwise binary
        metric scores:
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean. This does not take label imbalance into account. Classes
            are assumed to be uniformly distributed.
        ``'weighted'``:
            Calculate metrics for each label, taking into account the
            prevalence of the classes.

    Returns
    -------
    score : float
        Average of the pairwise binary metric scores.
    """
    check_consistent_length(y_true, y_score)
    y_true_unique = np.unique(y_true)
    n_classes = y_true_unique.shape[0]
    n_pairs = n_classes * (n_classes - 1) // 2
    pair_scores = np.empty(n_pairs)
    is_weighted = average == 'weighted'
    prevalence = np.empty(n_pairs) if is_weighted else None
    for ix, (a, b) in enumerate(combinations(y_true_unique, 2)):
        a_mask = y_true == a
        b_mask = y_true == b
        ab_mask = np.logical_or(a_mask, b_mask)
        if is_weighted:
            prevalence[ix] = np.average(ab_mask)
        a_true = a_mask[ab_mask]
        b_true = b_mask[ab_mask]
        a_true_score = binary_metric(a_true, y_score[ab_mask, a])
        b_true_score = binary_metric(b_true, y_score[ab_mask, b])
        pair_scores[ix] = (a_true_score + b_true_score) / 2
    return np.average(pair_scores, weights=prevalence)