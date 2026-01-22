from abc import ABCMeta, abstractmethod
from collections import defaultdict
import logging
import math
from gensim import interfaces, utils
import numpy as np
class OkapiBM25Model(BM25ABC):
    """The original Okapi BM25 scoring function of Robertson et al. [2]_.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.corpora import Dictionary
        >>> from gensim.models import OkapiBM25Model
        >>> from gensim.test.utils import common_texts
        >>>
        >>> dictionary = Dictionary(common_texts)  # fit dictionary
        >>> model = OkapiBM25Model(dictionary=dictionary)  # fit model
        >>>
        >>> corpus = [dictionary.doc2bow(line) for line in common_texts]  # convert corpus to BoW format
        >>> vector = model[corpus[0]]  # apply model to the first corpus document

    References
    ----------
    .. [2] Robertson S. E., Walker S., Jones S., Hancock-Beaulieu M. M., Gatford M. (1995).
       `Okapi at TREC-3 <http://research.microsoft.com/pubs/67649/okapi_trec3.pdf>`_.
       *NIST Special Publication 500-226*.

    """

    def __init__(self, corpus=None, dictionary=None, k1=1.5, b=0.75, epsilon=0.25):
        """Pre-compute the average length of a document and inverse term document frequencies,
        which will be used to weight term frequencies for the documents.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int) or None, optional
            An input corpus, which will be used to compute the average length of a document and
            inverse term document frequencies. If None, then `dictionary` will be used to compute
            the statistics. If both `corpus` and `dictionary` are None, the statistics will be left
            unintialized. Default is None.
        dictionary : :class:`~gensim.corpora.Dictionary`
            An input dictionary, which will be used to compute the average length of a document and
            inverse term document frequencies.  If None, then `corpus` will be used to compute the
            statistics. If both `corpus` and `dictionary` are None, the statistics will be left
            unintialized. Default is None.
        k1 : float
            A positive tuning parameter that determines the impact of the term frequency on its BM25
            weight. Singhal [5]_ suggests to set `k1` between 1.0 and 2.0. Default is 1.5.
        b : float
            A tuning parameter between 0.0 and 1.0 that determines the document length
            normalization: 1.0 corresponds to full document normalization, while 0.0 corresponds to
            no length normalization. Singhal [5]_ suggests to set `b` to 0.75, which is the default.
        epsilon : float
            A positive tuning parameter that lower-bounds an inverse document frequency.
            Defaults to 0.25.

        Attributes
        ----------
        k1 : float
            A positive tuning parameter that determines the impact of the term frequency on its BM25
            weight. Singhal [3]_ suggests to set `k1` between 1.0 and 2.0. Default is 1.5.
        b : float
            A tuning parameter between 0.0 and 1.0 that determines the document length
            normalization: 1.0 corresponds to full document normalization, while 0.0 corresponds to
            no length normalization. Singhal [3]_ suggests to set `b` to 0.75, which is the default.
        epsilon : float
            A positive tuning parameter that lower-bounds an inverse document frequency.
            Defaults to 0.25.

        References
        ----------
        .. [3] Singhal, A. (2001). `Modern information retrieval: A brief overview
           <http://singhal.info/ieee2001.pdf>`_. *IEEE Data Eng. Bull.*, 24(4), 35–43.

        """
        self.k1, self.b, self.epsilon = (k1, b, epsilon)
        super().__init__(corpus, dictionary)

    def precompute_idfs(self, dfs, num_docs):
        idf_sum = 0
        idfs = dict()
        negative_idfs = []
        for term_id, freq in dfs.items():
            idf = math.log(num_docs - freq + 0.5) - math.log(freq + 0.5)
            idfs[term_id] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(term_id)
        average_idf = idf_sum / len(idfs)
        eps = self.epsilon * average_idf
        for term_id in negative_idfs:
            idfs[term_id] = eps
        return idfs

    def get_term_weights(self, num_tokens, term_frequencies, idfs):
        term_weights = idfs * (term_frequencies * (self.k1 + 1) / (term_frequencies + self.k1 * (1 - self.b + self.b * num_tokens / self.avgdl)))
        return term_weights