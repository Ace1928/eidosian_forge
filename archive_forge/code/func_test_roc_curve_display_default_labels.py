import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.fixes import trapezoid
@pytest.mark.parametrize('roc_auc, estimator_name, expected_label', [(0.9, None, 'AUC = 0.90'), (None, 'my_est', 'my_est'), (0.8, 'my_est2', 'my_est2 (AUC = 0.80)')])
def test_roc_curve_display_default_labels(pyplot, roc_auc, estimator_name, expected_label):
    """Check the default labels used in the display."""
    fpr = np.array([0, 0.5, 1])
    tpr = np.array([0, 0.5, 1])
    disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=estimator_name).plot()
    assert disp.line_.get_label() == expected_label