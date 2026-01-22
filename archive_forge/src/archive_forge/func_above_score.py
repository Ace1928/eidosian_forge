import itertools as _itertools
from nltk.metrics import (
from nltk.metrics.spearman import ranks_from_scores, spearman_correlation
from nltk.probability import FreqDist
from nltk.util import ngrams
def above_score(self, score_fn, min_score):
    """Returns a sequence of ngrams, ordered by decreasing score, whose
        scores each exceed the given minimum score.
        """
    for ngram, score in self.score_ngrams(score_fn):
        if score > min_score:
            yield ngram
        else:
            break