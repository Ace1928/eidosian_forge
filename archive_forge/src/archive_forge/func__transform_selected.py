import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
def _transform_selected(X, transform, selected, copy=True):
    """Apply a transform function to portion of selected features.

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Dense array or sparse matrix.

    transform : callable
        A callable transform(X) -> X_transformed

    copy : boolean, optional
        Copy X even if it could be avoided.

    selected: "all", "auto" or array of indices or mask
        Specify which features to apply the transform to.

    Returns
    -------
    X : array or sparse matrix, shape=(n_samples, n_features_new)
    """
    if selected == 'all':
        return transform(X)
    if len(selected) == 0:
        return X
    X = check_array(X, accept_sparse='csc', force_all_finite=False)
    X_sel, X_not_sel, n_selected, n_features = _X_selected(X, selected)
    if n_selected == 0:
        return X
    elif n_selected == n_features:
        return transform(X)
    else:
        X_sel = transform(X_sel)
        if sparse.issparse(X_sel) or sparse.issparse(X_not_sel):
            return sparse.hstack((X_sel, X_not_sel), format='csr')
        else:
            return np.hstack((X_sel, X_not_sel))