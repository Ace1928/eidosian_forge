import logging
import itertools
import os
import heapq
import warnings
import numpy
import scipy.sparse
from gensim import interfaces, utils, matutils
class WmdSimilarity(interfaces.SimilarityABC):
    """Compute negative WMD similarity against a corpus of documents.

    Check out `the Gallery <https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html>`__
    for more examples.

    When using this code, please consider citing the following papers:

    * `Rémi Flamary et al. "POT: Python Optimal Transport"
      <https://jmlr.org/papers/v22/20-451.html>`_
    * `Matt Kusner et al. "From Word Embeddings To Document Distances"
      <http://proceedings.mlr.press/v37/kusnerb15.pdf>`_

    Example
    -------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import common_texts
        >>> from gensim.models import Word2Vec
        >>> from gensim.similarities import WmdSimilarity
        >>>
        >>> model = Word2Vec(common_texts, vector_size=20, min_count=1)  # train word-vectors
        >>>
        >>> index = WmdSimilarity(common_texts, model.wv)
        >>> # Make query.
        >>> query = ['trees']
        >>> sims = index[query]

    """

    def __init__(self, corpus, kv_model, num_best=None, chunksize=256):
        """

        Parameters
        ----------
        corpus: iterable of list of str
            A list of documents, each of which is a list of tokens.
        kv_model: :class:`~gensim.models.keyedvectors.KeyedVectors`
            A set of KeyedVectors
        num_best: int, optional
            Number of results to retrieve.
        chunksize : int, optional
            Size of chunk.

        """
        self.corpus = corpus
        self.wv = kv_model
        self.num_best = num_best
        self.chunksize = chunksize
        self.normalize = False
        self.index = numpy.arange(len(corpus))

    def __len__(self):
        """Get size of corpus."""
        return len(self.corpus)

    def get_similarities(self, query):
        """Get similarity between `query` and this index.

        Warnings
        --------
        Do not use this function directly; use the `self[query]` syntax instead.

        Parameters
        ----------
        query : {list of str, iterable of list of str}
            Document or collection of documents.

        Return
        ------
        :class:`numpy.ndarray`
            Similarity matrix.

        """
        if isinstance(query, numpy.ndarray):
            query = [self.corpus[i] for i in query]
        if not query or not isinstance(query[0], list):
            query = [query]
        n_queries = len(query)
        result = []
        for qidx in range(n_queries):
            qresult = [self.wv.wmdistance(document, query[qidx]) for document in self.corpus]
            qresult = numpy.array(qresult)
            qresult = 1.0 / (1.0 + qresult)
            result.append(qresult)
        if len(result) == 1:
            result = result[0]
        else:
            result = numpy.array(result)
        return result

    def __str__(self):
        return '%s<%i docs, %i features>' % (self.__class__.__name__, len(self), self.wv.vectors.shape[1])