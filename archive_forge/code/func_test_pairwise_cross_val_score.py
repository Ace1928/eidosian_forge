from re import escape
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from sklearn import datasets, svm
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.multiclass import (
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import (
from sklearn.utils._mocking import CheckingClassifier
from sklearn.utils._testing import assert_almost_equal, assert_array_equal
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import check_classification_targets, type_of_target
@pytest.mark.parametrize('MultiClassClassifier', [OneVsRestClassifier, OneVsOneClassifier])
def test_pairwise_cross_val_score(MultiClassClassifier):
    clf_precomputed = svm.SVC(kernel='precomputed')
    clf_notprecomputed = svm.SVC(kernel='linear')
    X, y = (iris.data, iris.target)
    multiclass_clf_notprecomputed = MultiClassClassifier(clf_notprecomputed)
    multiclass_clf_precomputed = MultiClassClassifier(clf_precomputed)
    linear_kernel = np.dot(X, X.T)
    score_not_precomputed = cross_val_score(multiclass_clf_notprecomputed, X, y, error_score='raise')
    score_precomputed = cross_val_score(multiclass_clf_precomputed, linear_kernel, y, error_score='raise')
    assert_array_equal(score_precomputed, score_not_precomputed)