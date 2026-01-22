from operator import methodcaller
from nltk.lm.api import Smoothing
from nltk.probability import ConditionalFreqDist
class WittenBell(Smoothing):
    """Witten-Bell smoothing."""

    def __init__(self, vocabulary, counter, **kwargs):
        super().__init__(vocabulary, counter, **kwargs)

    def alpha_gamma(self, word, context):
        alpha = self.counts[context].freq(word)
        gamma = self._gamma(context)
        return ((1.0 - gamma) * alpha, gamma)

    def _gamma(self, context):
        n_plus = _count_values_gt_zero(self.counts[context])
        return n_plus / (n_plus + self.counts[context].N())

    def unigram_score(self, word):
        return self.counts.unigrams.freq(word)