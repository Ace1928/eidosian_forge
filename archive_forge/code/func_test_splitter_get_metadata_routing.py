import re
import warnings
from itertools import combinations, combinations_with_replacement, permutations
import numpy as np
import pytest
from scipy import stats
from scipy.sparse import issparse
from scipy.special import comb
from sklearn import config_context
from sklearn.datasets import load_digits, make_classification
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
from sklearn.model_selection._split import (
from sklearn.svm import SVC
from sklearn.tests.metadata_routing_common import assert_request_is_empty
from sklearn.utils._array_api import (
from sklearn.utils._array_api import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
@pytest.mark.parametrize('cv', ALL_SPLITTERS, ids=[str(cv) for cv in ALL_SPLITTERS])
def test_splitter_get_metadata_routing(cv):
    """Check get_metadata_routing returns the correct MetadataRouter."""
    assert hasattr(cv, 'get_metadata_routing')
    metadata = cv.get_metadata_routing()
    if cv in GROUP_SPLITTERS:
        assert metadata.split.requests['groups'] is True
    elif cv in NO_GROUP_SPLITTERS:
        assert not metadata.split.requests
    assert_request_is_empty(metadata, exclude=['split'])