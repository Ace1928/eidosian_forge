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
@pytest.mark.parametrize('cv, expected', [(KFold(), True), (KFold(shuffle=True, random_state=123), True), (StratifiedKFold(), True), (StratifiedKFold(shuffle=True, random_state=123), True), (StratifiedGroupKFold(shuffle=True, random_state=123), True), (StratifiedGroupKFold(), True), (RepeatedKFold(random_state=123), True), (RepeatedStratifiedKFold(random_state=123), True), (ShuffleSplit(random_state=123), True), (GroupShuffleSplit(random_state=123), True), (StratifiedShuffleSplit(random_state=123), True), (GroupKFold(), True), (TimeSeriesSplit(), True), (LeaveOneOut(), True), (LeaveOneGroupOut(), True), (LeavePGroupsOut(n_groups=2), True), (LeavePOut(p=2), True), (KFold(shuffle=True, random_state=None), False), (KFold(shuffle=True, random_state=None), False), (StratifiedKFold(shuffle=True, random_state=np.random.RandomState(0)), False), (StratifiedKFold(shuffle=True, random_state=np.random.RandomState(0)), False), (RepeatedKFold(random_state=None), False), (RepeatedKFold(random_state=np.random.RandomState(0)), False), (RepeatedStratifiedKFold(random_state=None), False), (RepeatedStratifiedKFold(random_state=np.random.RandomState(0)), False), (ShuffleSplit(random_state=None), False), (ShuffleSplit(random_state=np.random.RandomState(0)), False), (GroupShuffleSplit(random_state=None), False), (GroupShuffleSplit(random_state=np.random.RandomState(0)), False), (StratifiedShuffleSplit(random_state=None), False), (StratifiedShuffleSplit(random_state=np.random.RandomState(0)), False)])
def test_yields_constant_splits(cv, expected):
    assert _yields_constant_splits(cv) == expected