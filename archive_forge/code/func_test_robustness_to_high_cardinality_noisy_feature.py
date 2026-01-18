import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.compose import ColumnTransformer
from sklearn.datasets import (
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler, scale
from sklearn.utils import parallel_backend
from sklearn.utils._testing import _convert_container
@pytest.mark.parametrize('n_jobs', [1, 2])
@pytest.mark.parametrize('max_samples', [0.5, 1.0])
def test_robustness_to_high_cardinality_noisy_feature(n_jobs, max_samples, seed=42):
    rng = np.random.RandomState(seed)
    n_repeats = 5
    n_samples = 1000
    n_classes = 5
    n_informative_features = 2
    n_noise_features = 1
    n_features = n_informative_features + n_noise_features
    classes = np.arange(n_classes)
    y = rng.choice(classes, size=n_samples)
    X = np.hstack([(y == c).reshape(-1, 1) for c in classes[:n_informative_features]])
    X = X.astype(np.float32)
    assert n_informative_features < n_classes
    X = np.concatenate([X, rng.randn(n_samples, n_noise_features)], axis=1)
    assert X.shape == (n_samples, n_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=rng)
    clf = RandomForestClassifier(n_estimators=5, random_state=rng)
    clf.fit(X_train, y_train)
    tree_importances = clf.feature_importances_
    informative_tree_importances = tree_importances[:n_informative_features]
    noisy_tree_importances = tree_importances[n_informative_features:]
    assert informative_tree_importances.max() < noisy_tree_importances.min()
    r = permutation_importance(clf, X_test, y_test, n_repeats=n_repeats, random_state=rng, n_jobs=n_jobs, max_samples=max_samples)
    assert r.importances.shape == (X.shape[1], n_repeats)
    informative_importances = r.importances_mean[:n_informative_features]
    noisy_importances = r.importances_mean[n_informative_features:]
    assert max(np.abs(noisy_importances)) > 1e-07
    assert noisy_importances.max() < 0.05
    assert informative_importances.min() > 0.15