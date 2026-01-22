import numbers
import operator
import time
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import partial, reduce
from itertools import product
import numpy as np
from numpy.ma import MaskedArray
from scipy.stats import rankdata
from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
from ..exceptions import NotFittedError
from ..metrics import check_scoring
from ..metrics._scorer import (
from ..utils import Bunch, check_random_state
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils._tags import _safe_tags
from ..utils.metadata_routing import (
from ..utils.metaestimators import available_if
from ..utils.parallel import Parallel, delayed
from ..utils.random import sample_without_replacement
from ..utils.validation import _check_method_params, check_is_fitted, indexable
from ._split import check_cv
from ._validation import (
def _get_routed_params_for_fit(self, params):
    """Get the parameters to be used for routing.

        This is a method instead of a snippet in ``fit`` since it's used twice,
        here in ``fit``, and in ``HalvingRandomSearchCV.fit``.
        """
    if _routing_enabled():
        routed_params = process_routing(self, 'fit', **params)
    else:
        params = params.copy()
        groups = params.pop('groups', None)
        routed_params = Bunch(estimator=Bunch(fit=params), splitter=Bunch(split={'groups': groups}), scorer=Bunch(score={}))
    return routed_params