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
def _validate_vocabulary(self):
    vocabulary = self.vocabulary
    if vocabulary is not None:
        if isinstance(vocabulary, set):
            vocabulary = sorted(vocabulary)
        if not isinstance(vocabulary, Mapping):
            vocab = {}
            for i, t in enumerate(vocabulary):
                if vocab.setdefault(t, i) != i:
                    msg = 'Duplicate term in vocabulary: %r' % t
                    raise ValueError(msg)
            vocabulary = vocab
        else:
            indices = set(vocabulary.values())
            if len(indices) != len(vocabulary):
                raise ValueError('Vocabulary contains repeated indices.')
            for i in range(len(vocabulary)):
                if i not in indices:
                    msg = "Vocabulary of size %d doesn't contain index %d." % (len(vocabulary), i)
                    raise ValueError(msg)
        if not vocabulary:
            raise ValueError('empty vocabulary passed to fit')
        self.fixed_vocabulary_ = True
        self.vocabulary_ = dict(vocabulary)
    else:
        self.fixed_vocabulary_ = False