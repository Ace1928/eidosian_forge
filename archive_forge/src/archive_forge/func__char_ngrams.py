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
def _char_ngrams(self, text_document):
    """Tokenize text_document into a sequence of character n-grams"""
    text_document = self._white_spaces.sub(' ', text_document)
    text_len = len(text_document)
    min_n, max_n = self.ngram_range
    if min_n == 1:
        ngrams = list(text_document)
        min_n += 1
    else:
        ngrams = []
    ngrams_append = ngrams.append
    for n in range(min_n, min(max_n + 1, text_len + 1)):
        for i in range(text_len - n + 1):
            ngrams_append(text_document[i:i + n])
    return ngrams