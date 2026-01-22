import sys
from collections import defaultdict
from nltk.classify.util import accuracy as eval_accuracy
from nltk.classify.util import apply_features
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.metrics import f_measure as eval_f_measure
from nltk.metrics import precision as eval_precision
from nltk.metrics import recall as eval_recall
from nltk.probability import FreqDist
def all_words(self, documents, labeled=None):
    """
        Return all words/tokens from the documents (with duplicates).

        :param documents: a list of (words, label) tuples.
        :param labeled: if `True`, assume that each document is represented by a
            (words, label) tuple: (list(str), str). If `False`, each document is
            considered as being a simple list of strings: list(str).
        :rtype: list(str)
        :return: A list of all words/tokens in `documents`.
        """
    all_words = []
    if labeled is None:
        labeled = documents and isinstance(documents[0], tuple)
    if labeled:
        for words, _sentiment in documents:
            all_words.extend(words)
    elif not labeled:
        for words in documents:
            all_words.extend(words)
    return all_words