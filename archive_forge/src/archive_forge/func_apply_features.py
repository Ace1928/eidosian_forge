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
def apply_features(self, documents, labeled=None):
    """
        Apply all feature extractor functions to the documents. This is a wrapper
        around `nltk.classify.util.apply_features`.

        If `labeled=False`, return featuresets as:
            [feature_func(doc) for doc in documents]
        If `labeled=True`, return featuresets as:
            [(feature_func(tok), label) for (tok, label) in toks]

        :param documents: a list of documents. `If labeled=True`, the method expects
            a list of (words, label) tuples.
        :rtype: LazyMap
        """
    return apply_features(self.extract_features, documents, labeled)