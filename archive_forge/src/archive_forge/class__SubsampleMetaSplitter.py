from abc import abstractmethod
from copy import deepcopy
from math import ceil, floor, log
from numbers import Integral, Real
import numpy as np
from ..base import _fit_context, is_classifier
from ..metrics._scorer import get_scorer_names
from ..utils import resample
from ..utils._param_validation import Interval, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.validation import _num_samples
from . import ParameterGrid, ParameterSampler
from ._search import BaseSearchCV
from ._split import _yields_constant_splits, check_cv
class _SubsampleMetaSplitter:
    """Splitter that subsamples a given fraction of the dataset"""

    def __init__(self, *, base_cv, fraction, subsample_test, random_state):
        self.base_cv = base_cv
        self.fraction = fraction
        self.subsample_test = subsample_test
        self.random_state = random_state

    def split(self, X, y, **kwargs):
        for train_idx, test_idx in self.base_cv.split(X, y, **kwargs):
            train_idx = resample(train_idx, replace=False, random_state=self.random_state, n_samples=int(self.fraction * len(train_idx)))
            if self.subsample_test:
                test_idx = resample(test_idx, replace=False, random_state=self.random_state, n_samples=int(self.fraction * len(test_idx)))
            yield (train_idx, test_idx)