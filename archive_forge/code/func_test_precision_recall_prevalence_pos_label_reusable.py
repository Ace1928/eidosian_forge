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
@pytest.mark.parametrize('constructor_name', ['from_estimator', 'from_predictions'])
def test_precision_recall_prevalence_pos_label_reusable(pyplot, constructor_name):
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)
    lr = LogisticRegression()
    y_pred = lr.fit(X, y).predict_proba(X)[:, 1]
    if constructor_name == 'from_estimator':
        display = PrecisionRecallDisplay.from_estimator(lr, X, y, plot_chance_level=False)
    else:
        display = PrecisionRecallDisplay.from_predictions(y, y_pred, plot_chance_level=False)
    assert display.chance_level_ is None
    import matplotlib as mpl
    display.plot(plot_chance_level=True)
    assert isinstance(display.chance_level_, mpl.lines.Line2D)