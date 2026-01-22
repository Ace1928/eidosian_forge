import itertools as _itertools
from nltk.metrics import (
from nltk.metrics.spearman import ranks_from_scores, spearman_correlation
from nltk.probability import FreqDist
from nltk.util import ngrams
@classmethod
def _build_new_documents(cls, documents, window_size, pad_left=False, pad_right=False, pad_symbol=None):
    """
        Pad the document with the place holder according to the window_size
        """
    padding = (pad_symbol,) * (window_size - 1)
    if pad_right:
        return _itertools.chain.from_iterable((_itertools.chain(doc, padding) for doc in documents))
    if pad_left:
        return _itertools.chain.from_iterable((_itertools.chain(padding, doc) for doc in documents))