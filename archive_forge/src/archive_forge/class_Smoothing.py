import random
import warnings
from abc import ABCMeta, abstractmethod
from bisect import bisect
from itertools import accumulate
from nltk.lm.counter import NgramCounter
from nltk.lm.util import log_base2
from nltk.lm.vocabulary import Vocabulary
class Smoothing(metaclass=ABCMeta):
    """Ngram Smoothing Interface

    Implements Chen & Goodman 1995's idea that all smoothing algorithms have
    certain features in common. This should ideally allow smoothing algorithms to
    work both with Backoff and Interpolation.
    """

    def __init__(self, vocabulary, counter):
        """
        :param vocabulary: The Ngram vocabulary object.
        :type vocabulary: nltk.lm.vocab.Vocabulary
        :param counter: The counts of the vocabulary items.
        :type counter: nltk.lm.counter.NgramCounter
        """
        self.vocab = vocabulary
        self.counts = counter

    @abstractmethod
    def unigram_score(self, word):
        raise NotImplementedError()

    @abstractmethod
    def alpha_gamma(self, word, context):
        raise NotImplementedError()