import numpy as np
import pytest
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsTransformer, RadiusNeighborsTransformer
from sklearn.neighbors._base import _is_sorted_by_data
from sklearn.utils._testing import assert_array_equal
def test_transformer_result():
    n_neighbors = 5
    n_samples_fit = 20
    n_queries = 18
    n_features = 10
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples_fit, n_features)
    X2 = rng.randn(n_queries, n_features)
    radius = np.percentile(euclidean_distances(X), 10)
    for mode in ['distance', 'connectivity']:
        add_one = mode == 'distance'
        nnt = KNeighborsTransformer(n_neighbors=n_neighbors, mode=mode)
        Xt = nnt.fit_transform(X)
        assert Xt.shape == (n_samples_fit, n_samples_fit)
        assert Xt.data.shape == (n_samples_fit * (n_neighbors + add_one),)
        assert Xt.format == 'csr'
        assert _is_sorted_by_data(Xt)
        X2t = nnt.transform(X2)
        assert X2t.shape == (n_queries, n_samples_fit)
        assert X2t.data.shape == (n_queries * (n_neighbors + add_one),)
        assert X2t.format == 'csr'
        assert _is_sorted_by_data(X2t)
    for mode in ['distance', 'connectivity']:
        add_one = mode == 'distance'
        nnt = RadiusNeighborsTransformer(radius=radius, mode=mode)
        Xt = nnt.fit_transform(X)
        assert Xt.shape == (n_samples_fit, n_samples_fit)
        assert not Xt.data.shape == (n_samples_fit * (n_neighbors + add_one),)
        assert Xt.format == 'csr'
        assert _is_sorted_by_data(Xt)
        X2t = nnt.transform(X2)
        assert X2t.shape == (n_queries, n_samples_fit)
        assert not X2t.data.shape == (n_queries * (n_neighbors + add_one),)
        assert X2t.format == 'csr'
        assert _is_sorted_by_data(X2t)