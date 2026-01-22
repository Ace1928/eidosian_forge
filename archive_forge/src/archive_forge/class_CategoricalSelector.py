import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.decomposition import PCA
from .one_hot_encoder import OneHotEncoder, auto_select_categorical_features, _X_selected
class CategoricalSelector(BaseEstimator, TransformerMixin):
    """Meta-transformer for selecting categorical features and transform them using OneHotEncoder.

    Parameters
    ----------

    threshold : int, default=10
        Maximum number of unique values per feature to consider the feature
        to be categorical.

    minimum_fraction: float, default=None
        Minimum fraction of unique values in a feature to consider the feature
        to be categorical.
    """

    def __init__(self, threshold=10, minimum_fraction=None):
        """Create a CategoricalSelector object."""
        self.threshold = threshold
        self.minimum_fraction = minimum_fraction

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged
        This method is just there to implement the usual API and hence
        work in pipelines.
        Parameters
        ----------
        X : array-like
        """
        X = check_array(X, accept_sparse='csr')
        return self

    def transform(self, X):
        """Select categorical features and transform them using OneHotEncoder.

        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components is the number of components.

        Returns
        -------
        array-like, {n_samples, n_components}
        """
        selected = auto_select_categorical_features(X, threshold=self.threshold)
        X_sel, _, n_selected, _ = _X_selected(X, selected)
        if n_selected == 0:
            raise ValueError('No categorical feature was found!')
        else:
            ohe = OneHotEncoder(categorical_features='all', sparse=False, minimum_fraction=self.minimum_fraction)
            return ohe.fit_transform(X_sel)