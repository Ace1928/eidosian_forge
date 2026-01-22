import itertools as _itertools
from nltk.metrics import (
from nltk.metrics.spearman import ranks_from_scores, spearman_correlation
from nltk.probability import FreqDist
from nltk.util import ngrams
class TrigramCollocationFinder(AbstractCollocationFinder):
    """A tool for the finding and ranking of trigram collocations or other
    association measures. It is often useful to use from_words() rather than
    constructing an instance directly.
    """
    default_ws = 3

    def __init__(self, word_fd, bigram_fd, wildcard_fd, trigram_fd):
        """Construct a TrigramCollocationFinder, given FreqDists for
        appearances of words, bigrams, two words with any word between them,
        and trigrams.
        """
        AbstractCollocationFinder.__init__(self, word_fd, trigram_fd)
        self.wildcard_fd = wildcard_fd
        self.bigram_fd = bigram_fd

    @classmethod
    def from_words(cls, words, window_size=3):
        """Construct a TrigramCollocationFinder for all trigrams in the given
        sequence.
        """
        if window_size < 3:
            raise ValueError('Specify window_size at least 3')
        wfd = FreqDist()
        wildfd = FreqDist()
        bfd = FreqDist()
        tfd = FreqDist()
        for window in ngrams(words, window_size, pad_right=True):
            w1 = window[0]
            if w1 is None:
                continue
            for w2, w3 in _itertools.combinations(window[1:], 2):
                wfd[w1] += 1
                if w2 is None:
                    continue
                bfd[w1, w2] += 1
                if w3 is None:
                    continue
                wildfd[w1, w3] += 1
                tfd[w1, w2, w3] += 1
        return cls(wfd, bfd, wildfd, tfd)

    def bigram_finder(self):
        """Constructs a bigram collocation finder with the bigram and unigram
        data from this finder. Note that this does not include any filtering
        applied to this finder.
        """
        return BigramCollocationFinder(self.word_fd, self.bigram_fd)

    def score_ngram(self, score_fn, w1, w2, w3):
        """Returns the score for a given trigram using the given scoring
        function.
        """
        n_all = self.N
        n_iii = self.ngram_fd[w1, w2, w3]
        if not n_iii:
            return
        n_iix = self.bigram_fd[w1, w2]
        n_ixi = self.wildcard_fd[w1, w3]
        n_xii = self.bigram_fd[w2, w3]
        n_ixx = self.word_fd[w1]
        n_xix = self.word_fd[w2]
        n_xxi = self.word_fd[w3]
        return score_fn(n_iii, (n_iix, n_ixi, n_xii), (n_ixx, n_xix, n_xxi), n_all)