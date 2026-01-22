from . import utils
from scipy import sparse
from sklearn import decomposition
from sklearn import random_projection
import numpy as np
import pandas as pd
import sklearn.base
import warnings
class InvertibleRandomProjection(random_projection.GaussianRandomProjection):
    """Gaussian random projection with an inverse transform using the pseudoinverse."""

    def __init__(self, n_components='auto', eps=0.3, orthogonalize=False, random_state=None):
        self.orthogonalize = orthogonalize
        super().__init__(n_components=n_components, eps=eps, random_state=random_state)

    @property
    def pseudoinverse(self):
        """Pseudoinverse of the random projection.

        This inverts the projection operation for any vector in the span of the
        random projection. For small enough `eps`, this should be close to the
        correct inverse.
        """
        try:
            return self._pseudoinverse
        except AttributeError:
            if self.orthogonalize:
                self._pseudoinverse = self.components_
            else:
                self._pseudoinverse = np.linalg.pinv(self.components_.T)
            return self._pseudoinverse

    def fit(self, X):
        super().fit(X)
        if self.orthogonalize:
            Q, _ = np.linalg.qr(self.components_.T)
            self.components_ = Q.T
        return self

    def inverse_transform(self, X):
        return X.dot(self.pseudoinverse)