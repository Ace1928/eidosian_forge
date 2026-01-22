import warnings
from math import log
from numbers import Real
import numpy as np
from scipy import sparse as sp
from ...utils._param_validation import Interval, StrOptions, validate_params
from ...utils.multiclass import type_of_target
from ...utils.validation import check_array, check_consistent_length
from ._expected_mutual_info_fast import expected_mutual_information
@validate_params({'labels_true': ['array-like'], 'labels_pred': ['array-like'], 'average_method': [StrOptions({'arithmetic', 'max', 'min', 'geometric'})]}, prefer_skip_nested_validation=True)
def adjusted_mutual_info_score(labels_true, labels_pred, *, average_method='arithmetic'):
    """Adjusted Mutual Information between two clusterings.

    Adjusted Mutual Information (AMI) is an adjustment of the Mutual
    Information (MI) score to account for chance. It accounts for the fact that
    the MI is generally higher for two clusterings with a larger number of
    clusters, regardless of whether there is actually more information shared.
    For two clusterings :math:`U` and :math:`V`, the AMI is given as::

        AMI(U, V) = [MI(U, V) - E(MI(U, V))] / [avg(H(U), H(V)) - E(MI(U, V))]

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching :math:`U` (``label_true``)
    with :math:`V` (``labels_pred``) will return the same score value. This can
    be useful to measure the agreement of two independent label assignments
    strategies on the same dataset when the real ground truth is not known.

    Be mindful that this function is an order of magnitude slower than other
    metrics, such as the Adjusted Rand Index.

    Read more in the :ref:`User Guide <mutual_info_score>`.

    Parameters
    ----------
    labels_true : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets, called :math:`U` in
        the above formula.

    labels_pred : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets, called :math:`V` in
        the above formula.

    average_method : {'min', 'geometric', 'arithmetic', 'max'}, default='arithmetic'
        How to compute the normalizer in the denominator.

        .. versionadded:: 0.20

        .. versionchanged:: 0.22
           The default value of ``average_method`` changed from 'max' to
           'arithmetic'.

    Returns
    -------
    ami: float (upperlimited by 1.0)
       The AMI returns a value of 1 when the two partitions are identical
       (ie perfectly matched). Random partitions (independent labellings) have
       an expected AMI around 0 on average hence can be negative. The value is
       in adjusted nats (based on the natural logarithm).

    See Also
    --------
    adjusted_rand_score : Adjusted Rand Index.
    mutual_info_score : Mutual Information (not adjusted for chance).

    References
    ----------
    .. [1] `Vinh, Epps, and Bailey, (2010). Information Theoretic Measures for
       Clusterings Comparison: Variants, Properties, Normalization and
       Correction for Chance, JMLR
       <http://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf>`_

    .. [2] `Wikipedia entry for the Adjusted Mutual Information
       <https://en.wikipedia.org/wiki/Adjusted_Mutual_Information>`_

    Examples
    --------

    Perfect labelings are both homogeneous and complete, hence have
    score 1.0::

      >>> from sklearn.metrics.cluster import adjusted_mutual_info_score
      >>> adjusted_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
      ... # doctest: +SKIP
      1.0
      >>> adjusted_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
      ... # doctest: +SKIP
      1.0

    If classes members are completely split across different clusters,
    the assignment is totally in-complete, hence the AMI is null::

      >>> adjusted_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])
      ... # doctest: +SKIP
      0.0
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    if classes.shape[0] == clusters.shape[0] == 1 or classes.shape[0] == clusters.shape[0] == 0:
        return 1.0
    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)
    emi = expected_mutual_information(contingency, n_samples)
    h_true, h_pred = (entropy(labels_true), entropy(labels_pred))
    normalizer = _generalized_average(h_true, h_pred, average_method)
    denominator = normalizer - emi
    if denominator < 0:
        denominator = min(denominator, -np.finfo('float64').eps)
    else:
        denominator = max(denominator, np.finfo('float64').eps)
    ami = (mi - emi) / denominator
    return ami