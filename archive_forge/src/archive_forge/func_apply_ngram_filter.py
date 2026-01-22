import itertools as _itertools
from nltk.metrics import (
from nltk.metrics.spearman import ranks_from_scores, spearman_correlation
from nltk.probability import FreqDist
from nltk.util import ngrams
def apply_ngram_filter(self, fn):
    """Removes candidate ngrams (w1, w2, ...) where fn(w1, w2, ...)
        evaluates to True.
        """
    self._apply_filter(lambda ng, f: fn(*ng))