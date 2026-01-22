from operator import methodcaller
from nltk.lm.api import Smoothing
from nltk.probability import ConditionalFreqDist
class KneserNey(Smoothing):
    """Kneser-Ney Smoothing.

    This is an extension of smoothing with a discount.

    Resources:
    - https://pages.ucsd.edu/~rlevy/lign256/winter2008/kneser_ney_mini_example.pdf
    - https://www.youtube.com/watch?v=ody1ysUTD7o
    - https://medium.com/@dennyc/a-simple-numerical-example-for-kneser-ney-smoothing-nlp-4600addf38b8
    - https://www.cl.uni-heidelberg.de/courses/ss15/smt/scribe6.pdf
    - https://www-i6.informatik.rwth-aachen.de/publications/download/951/Kneser-ICASSP-1995.pdf
    """

    def __init__(self, vocabulary, counter, order, discount=0.1, **kwargs):
        super().__init__(vocabulary, counter, **kwargs)
        self.discount = discount
        self._order = order

    def unigram_score(self, word):
        word_continuation_count, total_count = self._continuation_counts(word)
        return word_continuation_count / total_count

    def alpha_gamma(self, word, context):
        prefix_counts = self.counts[context]
        word_continuation_count, total_count = (prefix_counts[word], prefix_counts.N()) if len(context) + 1 == self._order else self._continuation_counts(word, context)
        alpha = max(word_continuation_count - self.discount, 0.0) / total_count
        gamma = self.discount * _count_values_gt_zero(prefix_counts) / total_count
        return (alpha, gamma)

    def _continuation_counts(self, word, context=tuple()):
        """Count continuations that end with context and word.

        Continuations track unique ngram "types", regardless of how many
        instances were observed for each "type".
        This is different than raw ngram counts which track number of instances.
        """
        higher_order_ngrams_with_context = (counts for prefix_ngram, counts in self.counts[len(context) + 2].items() if prefix_ngram[1:] == context)
        higher_order_ngrams_with_word_count, total = (0, 0)
        for counts in higher_order_ngrams_with_context:
            higher_order_ngrams_with_word_count += int(counts[word] > 0)
            total += _count_values_gt_zero(counts)
        return (higher_order_ngrams_with_word_count, total)