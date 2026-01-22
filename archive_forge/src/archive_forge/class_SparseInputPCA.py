from . import utils
from scipy import sparse
from sklearn import decomposition
from sklearn import random_projection
import numpy as np
import pandas as pd
import sklearn.base
import warnings
class SparseInputPCA(sklearn.base.BaseEstimator):
    """Calculate PCA using random projections to handle sparse matrices.

    Uses the Johnson-Lindenstrauss Lemma to determine the number of
    dimensions of random projections prior to subtracting the mean.

    Parameters
    ----------
    n_components : int, optional (default: 2)
        Number of components to keep.
    eps : strictly positive float, optional (default=0.15)
        Parameter to control the quality of the embedding according to the
        Johnson-Lindenstrauss lemma when n_components is set to ‘auto’.
        Smaller values lead to more accurate embeddings but higher
        computational and memory costs
    method : {'svd', 'orth_rproj', 'rproj'}, optional (default: 'svd')
        Dimensionality reduction method applied prior to mean centering.
        The method choice affects accuracy (`svd` > `orth_rproj` > `rproj`)
        comes with increased computational cost (but not memory.)
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by np.random.
    kwargs
        Additional keyword arguments for `sklearn.decomposition.PCA`
    """

    def __init__(self, n_components=2, eps=0.3, random_state=None, method='svd', **kwargs):
        self.pca_op = decomposition.PCA(n_components=n_components, random_state=random_state)
        if method == 'svd':
            self.proj_op = AutomaticDimensionSVD(eps=eps, random_state=random_state)
        elif method == 'orth_rproj':
            self.proj_op = InvertibleRandomProjection(eps=eps, random_state=random_state, orthogonalize=True)
        elif method == 'rproj':
            self.proj_op = InvertibleRandomProjection(eps=eps, random_state=random_state, orthogonalize=False)
        else:
            raise ValueError("Expected `method` in ['svd', 'orth_rproj', 'rproj']. Got '{}'".format(method))

    @property
    def singular_values_(self):
        """Singular values of the PCA decomposition."""
        return self.pca_op.singular_values_

    @property
    def explained_variance_(self):
        """The amount of variance explained by each of the selected components."""
        return self.pca_op.explained_variance_

    @property
    def explained_variance_ratio_(self):
        """Percentage of variance explained by each of the selected components.

        The sum of the ratios is equal to 1.0.
        If n_components is `None` then the number of components grows as`eps`
        gets smaller.
        """
        return self.pca_op.explained_variance_ratio_

    @property
    def components_(self):
        """Principal axes in feature space, representing directions of maximum variance.

        The components are sorted by explained variance.
        """
        return self.proj_op.inverse_transform(self.pca_op.components_)

    def _fit(self, X):
        self.proj_op.fit(X)
        X_proj = self.proj_op.transform(X)
        self.pca_op.fit(X_proj)
        return X_proj

    def fit(self, X):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
        """
        self._fit(X)
        return self

    def transform(self, X):
        """Apply dimensionality reduction to X.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)

        Returns
        -------
        X_new : array-like, shape=(n_samples, n_components)
        """
        X_proj = self.proj_op.transform(X)
        X_pca = self.pca_op.transform(X_proj)
        return X_pca

    def fit_transform(self, X):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)

        Returns
        -------
        X_new : array-like, shape=(n_samples, n_components)
        """
        X_proj = self._fit(X)
        X_pca = self.pca_op.transform(X_proj)
        return X_pca

    def inverse_transform(self, X):
        """Transform data back to its original space.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_components)

        Returns
        -------
        X_new : array-like, shape=(n_samples, n_features)
        """
        X_proj = self.pca_op.inverse_transform(X)
        X_ambient = self.proj_op.inverse_transform(X_proj)
        return X_ambient