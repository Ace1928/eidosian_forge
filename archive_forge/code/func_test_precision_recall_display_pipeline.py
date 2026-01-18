from collections import Counter
import numpy as np
import pytest
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.fixes import trapezoid
@pytest.mark.parametrize('clf', [make_pipeline(StandardScaler(), LogisticRegression()), make_pipeline(make_column_transformer((StandardScaler(), [0, 1])), LogisticRegression())])
def test_precision_recall_display_pipeline(pyplot, clf):
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)
    with pytest.raises(NotFittedError):
        PrecisionRecallDisplay.from_estimator(clf, X, y)
    clf.fit(X, y)
    display = PrecisionRecallDisplay.from_estimator(clf, X, y)
    assert display.estimator_name == clf.__class__.__name__