import numpy as np
from scipy.optimize import linear_sum_assignment
from ...utils._param_validation import StrOptions, validate_params
from ...utils.validation import check_array, check_consistent_length
The similarity of two sets of biclusters.

    Similarity between individual biclusters is computed. Then the
    best matching between sets is found using the Hungarian algorithm.
    The final score is the sum of similarities divided by the size of
    the larger set.

    Read more in the :ref:`User Guide <biclustering>`.

    Parameters
    ----------
    a : tuple (rows, columns)
        Tuple of row and column indicators for a set of biclusters.

    b : tuple (rows, columns)
        Another set of biclusters like ``a``.

    similarity : 'jaccard' or callable, default='jaccard'
        May be the string "jaccard" to use the Jaccard coefficient, or
        any function that takes four arguments, each of which is a 1d
        indicator vector: (a_rows, a_columns, b_rows, b_columns).

    Returns
    -------
    consensus_score : float
       Consensus score, a non-negative value, sum of similarities
       divided by size of larger set.

    References
    ----------

    * Hochreiter, Bodenhofer, et. al., 2010. `FABIA: factor analysis
      for bicluster acquisition
      <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2881408/>`__.

    Examples
    --------
    >>> from sklearn.metrics import consensus_score
    >>> a = ([[True, False], [False, True]], [[False, True], [True, False]])
    >>> b = ([[False, True], [True, False]], [[True, False], [False, True]])
    >>> consensus_score(a, b, similarity='jaccard')
    1.0
    