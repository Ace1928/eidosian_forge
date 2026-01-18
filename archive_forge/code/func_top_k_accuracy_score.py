import warnings
from functools import partial
from numbers import Integral, Real
import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.stats import rankdata
from ..exceptions import UndefinedMetricWarning
from ..preprocessing import label_binarize
from ..utils import (
from ..utils._encode import _encode, _unique
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import stable_cumsum
from ..utils.fixes import trapezoid
from ..utils.multiclass import type_of_target
from ..utils.sparsefuncs import count_nonzero
from ..utils.validation import _check_pos_label_consistency, _check_sample_weight
from ._base import _average_binary_score, _average_multiclass_ovo_score
@validate_params({'y_true': ['array-like'], 'y_score': ['array-like'], 'k': [Interval(Integral, 1, None, closed='left')], 'normalize': ['boolean'], 'sample_weight': ['array-like', None], 'labels': ['array-like', None]}, prefer_skip_nested_validation=True)
def top_k_accuracy_score(y_true, y_score, *, k=2, normalize=True, sample_weight=None, labels=None):
    """Top-k Accuracy classification score.

    This metric computes the number of times where the correct label is among
    the top `k` labels predicted (ranked by predicted scores). Note that the
    multilabel case isn't covered here.

    Read more in the :ref:`User Guide <top_k_accuracy_score>`

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.

    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores. These can be either probability estimates or
        non-thresholded decision values (as returned by
        :term:`decision_function` on some classifiers).
        The binary case expects scores with shape (n_samples,) while the
        multiclass case expects scores with shape (n_samples, n_classes).
        In the multiclass case, the order of the class scores must
        correspond to the order of ``labels``, if provided, or else to
        the numerical or lexicographical order of the labels in ``y_true``.
        If ``y_true`` does not contain all the labels, ``labels`` must be
        provided.

    k : int, default=2
        Number of most likely outcomes considered to find the correct label.

    normalize : bool, default=True
        If `True`, return the fraction of correctly classified samples.
        Otherwise, return the number of correctly classified samples.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If `None`, all samples are given the same weight.

    labels : array-like of shape (n_classes,), default=None
        Multiclass only. List of labels that index the classes in ``y_score``.
        If ``None``, the numerical or lexicographical order of the labels in
        ``y_true`` is used. If ``y_true`` does not contain all the labels,
        ``labels`` must be provided.

    Returns
    -------
    score : float
        The top-k accuracy score. The best performance is 1 with
        `normalize == True` and the number of samples with
        `normalize == False`.

    See Also
    --------
    accuracy_score : Compute the accuracy score. By default, the function will
        return the fraction of correct predictions divided by the total number
        of predictions.

    Notes
    -----
    In cases where two or more labels are assigned equal predicted scores,
    the labels with the highest indices will be chosen first. This might
    impact the result if the correct label falls after the threshold because
    of that.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import top_k_accuracy_score
    >>> y_true = np.array([0, 1, 2, 2])
    >>> y_score = np.array([[0.5, 0.2, 0.2],  # 0 is in top 2
    ...                     [0.3, 0.4, 0.2],  # 1 is in top 2
    ...                     [0.2, 0.4, 0.3],  # 2 is in top 2
    ...                     [0.7, 0.2, 0.1]]) # 2 isn't in top 2
    >>> top_k_accuracy_score(y_true, y_score, k=2)
    0.75
    >>> # Not normalizing gives the number of "correctly" classified samples
    >>> top_k_accuracy_score(y_true, y_score, k=2, normalize=False)
    3
    """
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_true = column_or_1d(y_true)
    y_type = type_of_target(y_true, input_name='y_true')
    if y_type == 'binary' and labels is not None and (len(labels) > 2):
        y_type = 'multiclass'
    if y_type not in {'binary', 'multiclass'}:
        raise ValueError(f"y type must be 'binary' or 'multiclass', got '{y_type}' instead.")
    y_score = check_array(y_score, ensure_2d=False)
    if y_type == 'binary':
        if y_score.ndim == 2 and y_score.shape[1] != 1:
            raise ValueError(f'`y_true` is binary while y_score is 2d with {y_score.shape[1]} classes. If `y_true` does not contain all the labels, `labels` must be provided.')
        y_score = column_or_1d(y_score)
    check_consistent_length(y_true, y_score, sample_weight)
    y_score_n_classes = y_score.shape[1] if y_score.ndim == 2 else 2
    if labels is None:
        classes = _unique(y_true)
        n_classes = len(classes)
        if n_classes != y_score_n_classes:
            raise ValueError(f"Number of classes in 'y_true' ({n_classes}) not equal to the number of classes in 'y_score' ({y_score_n_classes}).You can provide a list of all known classes by assigning it to the `labels` parameter.")
    else:
        labels = column_or_1d(labels)
        classes = _unique(labels)
        n_labels = len(labels)
        n_classes = len(classes)
        if n_classes != n_labels:
            raise ValueError("Parameter 'labels' must be unique.")
        if not np.array_equal(classes, labels):
            raise ValueError("Parameter 'labels' must be ordered.")
        if n_classes != y_score_n_classes:
            raise ValueError(f"Number of given labels ({n_classes}) not equal to the number of classes in 'y_score' ({y_score_n_classes}).")
        if len(np.setdiff1d(y_true, classes)):
            raise ValueError("'y_true' contains labels not in parameter 'labels'.")
    if k >= n_classes:
        warnings.warn(f"'k' ({k}) greater than or equal to 'n_classes' ({n_classes}) will result in a perfect score and is therefore meaningless.", UndefinedMetricWarning)
    y_true_encoded = _encode(y_true, uniques=classes)
    if y_type == 'binary':
        if k == 1:
            threshold = 0.5 if y_score.min() >= 0 and y_score.max() <= 1 else 0
            y_pred = (y_score > threshold).astype(np.int64)
            hits = y_pred == y_true_encoded
        else:
            hits = np.ones_like(y_score, dtype=np.bool_)
    elif y_type == 'multiclass':
        sorted_pred = np.argsort(y_score, axis=1, kind='mergesort')[:, ::-1]
        hits = (y_true_encoded == sorted_pred[:, :k].T).any(axis=0)
    if normalize:
        return np.average(hits, weights=sample_weight)
    elif sample_weight is None:
        return np.sum(hits)
    else:
        return np.dot(hits, sample_weight)