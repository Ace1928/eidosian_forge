import pytest
from sklearn.base import BaseEstimator
from sklearn.utils._tags import (
@pytest.mark.parametrize('estimator, err_msg', [(BaseEstimator(), 'The key xxx is not defined in _get_tags'), (NoTagsEstimator(), 'The key xxx is not defined in _DEFAULT_TAGS')])
def test_safe_tags_error(estimator, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        _safe_tags(estimator, key='xxx')