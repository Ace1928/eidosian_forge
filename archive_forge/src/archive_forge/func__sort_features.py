import array
import re
import unicodedata
import warnings
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from numbers import Integral
from operator import itemgetter
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin, _fit_context
from ..exceptions import NotFittedError
from ..preprocessing import normalize
from ..utils import _IS_32BIT
from ..utils._param_validation import HasMethods, Interval, RealNotInt, StrOptions
from ..utils.validation import FLOAT_DTYPES, check_array, check_is_fitted
from ._hash import FeatureHasher
from ._stop_words import ENGLISH_STOP_WORDS
def _sort_features(self, X, vocabulary):
    """Sort features by name

        Returns a reordered matrix and modifies the vocabulary in place
        """
    sorted_features = sorted(vocabulary.items())
    map_index = np.empty(len(sorted_features), dtype=X.indices.dtype)
    for new_val, (term, old_val) in enumerate(sorted_features):
        vocabulary[term] = new_val
        map_index[old_val] = new_val
    X.indices = map_index.take(X.indices, mode='clip')
    return X