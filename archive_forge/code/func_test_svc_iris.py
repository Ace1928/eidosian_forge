import numpy as np
import pytest
from scipy import sparse
from sklearn import base, datasets, linear_model, svm
from sklearn.datasets import load_digits, make_blobs, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm.tests import test_svm
from sklearn.utils._testing import (
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.fixes import (
@skip_if_32bit
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
@pytest.mark.parametrize('kernel', ['linear', 'poly', 'rbf'])
def test_svc_iris(csr_container, kernel):
    iris_data_sp = csr_container(iris.data)
    sp_clf = svm.SVC(kernel=kernel).fit(iris_data_sp, iris.target)
    clf = svm.SVC(kernel=kernel).fit(iris.data, iris.target)
    assert_allclose(clf.support_vectors_, sp_clf.support_vectors_.toarray())
    assert_allclose(clf.dual_coef_, sp_clf.dual_coef_.toarray())
    assert_allclose(clf.predict(iris.data), sp_clf.predict(iris_data_sp))
    if kernel == 'linear':
        assert_allclose(clf.coef_, sp_clf.coef_.toarray())