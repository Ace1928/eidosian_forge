import numpy as np
import pytest
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf, ShrunkCovariance, ledoit_wolf
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import (
from sklearn.preprocessing import StandardScaler
from sklearn.utils import _IS_WASM, check_random_state
from sklearn.utils._testing import (
class StandardizedLedoitWolf:

    def fit(self, X):
        sc = StandardScaler()
        X_sc = sc.fit_transform(X)
        s = ledoit_wolf(X_sc)[0]
        s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
        self.covariance_ = s