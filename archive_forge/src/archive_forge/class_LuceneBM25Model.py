from abc import ABCMeta, abstractmethod
from collections import defaultdict
import logging
import math
from gensim import interfaces, utils
import numpy as np
class LuceneBM25Model(BM25ABC):
    """The scoring function of Apache Lucene 8 [4]_.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.corpora import Dictionary
        >>> from gensim.models import LuceneBM25Model
        >>> from gensim.test.utils import common_texts
        >>>
        >>> dictionary = Dictionary(common_texts)  # fit dictionary
        >>> corpus = [dictionary.doc2bow(line) for line in common_texts]  # convert corpus to BoW format
        >>>
        >>> model = LuceneBM25Model(dictionary=dictionary)  # fit model
        >>> vector = model[corpus[0]]  # apply model to the first corpus document

    References
    ----------
    .. [4] Kamphuis, C., de Vries, A. P., Boytsov, L., Lin, J. (2020). Which
       BM25 Do You Mean? `A Large-Scale Reproducibility Study of Scoring Variants
       <https://doi.org/10.1007/978-3-030-45442-5_4>`_. In: Advances in Information Retrieval.
       28–34.

    """

    def __init__(self, corpus=None, dictionary=None, k1=1.5, b=0.75):
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

        Attributes
        ----------
        k1 : float
            A positive tuning parameter that determines the impact of the term frequency on its BM25
            weight. Singhal [3]_ suggests to set `k1` between 1.0 and 2.0. Default is 1.5.
        b : float
            A tuning parameter between 0.0 and 1.0 that determines the document length
            normalization: 1.0 corresponds to full document normalization, while 0.0 corresponds to
            no length normalization. Singhal [3]_ suggests to set `b` to 0.75, which is the default.

        """
        self.k1, self.b = (k1, b)
        super().__init__(corpus, dictionary)

    def precompute_idfs(self, dfs, num_docs):
        idfs = dict()
        for term_id, freq in dfs.items():
            idf = math.log(num_docs + 1.0) - math.log(freq + 0.5)
            idfs[term_id] = idf
        return idfs

    def get_term_weights(self, num_tokens, term_frequencies, idfs):
        term_weights = idfs * (term_frequencies / (term_frequencies + self.k1 * (1 - self.b + self.b * num_tokens / self.avgdl)))
        return term_weights