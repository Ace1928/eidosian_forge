import re
import warnings
from functools import partial
from itertools import chain, permutations, product
import numpy as np
import pytest
from scipy import linalg
from scipy.spatial.distance import hamming as sp_hamming
from scipy.stats import bernoulli
from sklearn import datasets, svm
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
from sklearn.metrics._classification import _check_targets
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.extmath import _nanaverage
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import check_random_state
def test_precision_recall_f_unused_pos_label():
    msg = "Note that pos_label \\(set to 2\\) is ignored when average != 'binary' \\(got 'macro'\\). You may use labels=\\[pos_label\\] to specify a single positive class."
    with pytest.warns(UserWarning, match=msg):
        precision_recall_fscore_support([1, 2, 1], [1, 2, 2], pos_label=2, average='macro')