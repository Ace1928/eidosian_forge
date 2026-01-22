import copy
import numbers
from abc import ABCMeta, abstractmethod
from math import ceil
from numbers import Integral, Real
import numpy as np
from scipy.sparse import issparse
from ..base import (
from ..utils import Bunch, check_random_state, compute_sample_weight
from ..utils._param_validation import Hidden, Interval, RealNotInt, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.validation import (
from . import _criterion, _splitter, _tree
from ._criterion import Criterion
from ._splitter import Splitter
from ._tree import (
from ._utils import _any_isnan_axis0
def _support_missing_values(self, X):
    return not issparse(X) and self._get_tags()['allow_nan'] and (self.monotonic_cst is None)