import itertools as _itertools
from nltk.metrics import (
from nltk.metrics.spearman import ranks_from_scores, spearman_correlation
from nltk.probability import FreqDist
from nltk.util import ngrams
class BigramCollocationFinder(AbstractCollocationFinder):
    """A tool for the finding and ranking of bigram collocations or other
    association measures. It is often useful to use from_words() rather than
    constructing an instance directly.
    """
    default_ws = 2

    def __init__(self, word_fd, bigram_fd, window_size=2):
        """Construct a BigramCollocationFinder, given FreqDists for
        appearances of words and (possibly non-contiguous) bigrams.
        """
        AbstractCollocationFinder.__init__(self, word_fd, bigram_fd)
        self.window_size = window_size

    @classmethod
    def from_words(cls, words, window_size=2):
        """Construct a BigramCollocationFinder for all bigrams in the given
        sequence.  When window_size > 2, count non-contiguous bigrams, in the
        style of Church and Hanks's (1990) association ratio.
        """
        wfd = FreqDist()
        bfd = FreqDist()
        if window_size < 2:
            raise ValueError('Specify window_size at least 2')
        for window in ngrams(words, window_size, pad_right=True):
            w1 = window[0]
            if w1 is None:
                continue
            wfd[w1] += 1
            for w2 in window[1:]:
                if w2 is not None:
                    bfd[w1, w2] += 1
        return cls(wfd, bfd, window_size=window_size)

    def score_ngram(self, score_fn, w1, w2):
        """Returns the score for a given bigram using the given scoring
        function.  Following Church and Hanks (1990), counts are scaled by
        a factor of 1/(window_size - 1).
        """
        n_all = self.N
        n_ii = self.ngram_fd[w1, w2] / (self.window_size - 1.0)
        if not n_ii:
            return
        n_ix = self.word_fd[w1]
        n_xi = self.word_fd[w2]
        return score_fn(n_ii, (n_ix, n_xi), n_all)