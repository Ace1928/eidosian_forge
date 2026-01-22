import numpy as np
from scipy.optimize import linear_sum_assignment
from ...utils._param_validation import StrOptions, validate_params
from ...utils.validation import check_array, check_consistent_length
def _pairwise_similarity(a, b, similarity):
    """Computes pairwise similarity matrix.

    result[i, j] is the Jaccard coefficient of a's bicluster i and b's
    bicluster j.

    """
    a_rows, a_cols, b_rows, b_cols = _check_rows_and_columns(a, b)
    n_a = a_rows.shape[0]
    n_b = b_rows.shape[0]
    result = np.array([[similarity(a_rows[i], a_cols[i], b_rows[j], b_cols[j]) for j in range(n_b)] for i in range(n_a)])
    return result