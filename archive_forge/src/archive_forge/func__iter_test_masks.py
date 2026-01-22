import numbers
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from inspect import signature
from itertools import chain, combinations
from math import ceil, floor
import numpy as np
from scipy.special import comb
from ..utils import (
from ..utils._param_validation import Interval, RealNotInt, validate_params
from ..utils.metadata_routing import _MetadataRequester
from ..utils.multiclass import type_of_target
from ..utils.validation import _num_samples, check_array, column_or_1d
def _iter_test_masks(self):
    """Generates boolean masks corresponding to test sets."""
    for f in self.unique_folds:
        test_index = np.where(self.test_fold == f)[0]
        test_mask = np.zeros(len(self.test_fold), dtype=bool)
        test_mask[test_index] = True
        yield test_mask