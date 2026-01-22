from operator import methodcaller
from nltk.lm.api import Smoothing
from nltk.probability import ConditionalFreqDist
class AbsoluteDiscounting(Smoothing):
    """Smoothing with absolute discount."""

    def __init__(self, vocabulary, counter, discount=0.75, **kwargs):
        super().__init__(vocabulary, counter, **kwargs)
        self.discount = discount

    def alpha_gamma(self, word, context):
        alpha = max(self.counts[context][word] - self.discount, 0) / self.counts[context].N()
        gamma = self._gamma(context)
        return (alpha, gamma)

    def _gamma(self, context):
        n_plus = _count_values_gt_zero(self.counts[context])
        return self.discount * n_plus / self.counts[context].N()

    def unigram_score(self, word):
        return self.counts.unigrams.freq(word)