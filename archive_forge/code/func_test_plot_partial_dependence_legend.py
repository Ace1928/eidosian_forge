import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats.mstats import mquantiles
from sklearn.compose import make_column_transformer
from sklearn.datasets import (
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._testing import _convert_container
def test_plot_partial_dependence_legend(pyplot):
    pd = pytest.importorskip('pandas')
    X = pd.DataFrame({'col_A': ['A', 'B', 'C'], 'col_B': [1, 0, 2], 'col_C': ['C', 'B', 'A']})
    y = np.array([1.2, 0.5, 0.45]).T
    categorical_features = ['col_A', 'col_C']
    preprocessor = make_column_transformer((OneHotEncoder(), categorical_features))
    model = make_pipeline(preprocessor, LinearRegression())
    model.fit(X, y)
    disp = PartialDependenceDisplay.from_estimator(model, X, features=['col_B', 'col_C'], categorical_features=categorical_features, kind=['both', 'average'])
    legend_text = disp.axes_[0, 0].get_legend().get_texts()
    assert len(legend_text) == 1
    assert legend_text[0].get_text() == 'average'
    assert disp.axes_[0, 1].get_legend() is None