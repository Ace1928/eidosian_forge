import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
def _matrix_adjust(self, X):
    """Adjust all values in X to encode for NaNs and infinities in the data.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_feature)
            Input array of type int.

        Returns
        -------
        X : array-like, shape=(n_samples, n_feature)
            Input array without any NaNs or infinities.
        """
    data_matrix = X.data if sparse.issparse(X) else X
    data_matrix += len(SPARSE_ENCODINGS) + 1
    data_matrix[~np.isfinite(data_matrix)] = SPARSE_ENCODINGS['NAN']
    return X