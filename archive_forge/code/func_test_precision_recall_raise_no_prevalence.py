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
def test_precision_recall_raise_no_prevalence(pyplot):
    precision = np.array([1, 0.5, 0])
    recall = np.array([0, 0.5, 1])
    display = PrecisionRecallDisplay(precision, recall)
    msg = 'You must provide prevalence_pos_label when constructing the PrecisionRecallDisplay object in order to plot the chance level line. Alternatively, you may use PrecisionRecallDisplay.from_estimator or PrecisionRecallDisplay.from_predictions to automatically set prevalence_pos_label'
    with pytest.raises(ValueError, match=msg):
        display.plot(plot_chance_level=True)