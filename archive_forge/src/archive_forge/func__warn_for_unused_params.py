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
def _warn_for_unused_params(self):
    if self.tokenizer is not None and self.token_pattern is not None:
        warnings.warn("The parameter 'token_pattern' will not be used since 'tokenizer' is not None'")
    if self.preprocessor is not None and callable(self.analyzer):
        warnings.warn("The parameter 'preprocessor' will not be used since 'analyzer' is callable'")
    if self.ngram_range != (1, 1) and self.ngram_range is not None and callable(self.analyzer):
        warnings.warn("The parameter 'ngram_range' will not be used since 'analyzer' is callable'")
    if self.analyzer != 'word' or callable(self.analyzer):
        if self.stop_words is not None:
            warnings.warn("The parameter 'stop_words' will not be used since 'analyzer' != 'word'")
        if self.token_pattern is not None and self.token_pattern != '(?u)\\b\\w\\w+\\b':
            warnings.warn("The parameter 'token_pattern' will not be used since 'analyzer' != 'word'")
        if self.tokenizer is not None:
            warnings.warn("The parameter 'tokenizer' will not be used since 'analyzer' != 'word'")