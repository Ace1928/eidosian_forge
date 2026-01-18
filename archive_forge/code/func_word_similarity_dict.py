import re
import sys
from collections import Counter, defaultdict, namedtuple
from functools import reduce
from math import log
from nltk.collocations import BigramCollocationFinder
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.metrics import BigramAssocMeasures, f_measure
from nltk.probability import ConditionalFreqDist as CFD
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize
from nltk.util import LazyConcatenation, tokenwrap
def word_similarity_dict(self, word):
    """
        Return a dictionary mapping from words to 'similarity scores,'
        indicating how often these two words occur in the same
        context.
        """
    word = self._key(word)
    word_contexts = set(self._word_to_contexts[word])
    scores = {}
    for w, w_contexts in self._word_to_contexts.items():
        scores[w] = f_measure(word_contexts, set(w_contexts))
    return scores