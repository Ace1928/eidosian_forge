import logging
import itertools
import os
import heapq
import warnings
import numpy
import scipy.sparse
from gensim import interfaces, utils, matutils
class MatrixSimilarity(interfaces.SimilarityABC):
    """Compute cosine similarity against a corpus of documents by storing the index matrix in memory.

    Unless the entire matrix fits into main memory, use :class:`~gensim.similarities.docsim.Similarity` instead.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import common_corpus, common_dictionary
        >>> from gensim.similarities import MatrixSimilarity
        >>>
        >>> query = [(1, 2), (5, 4)]
        >>> index = MatrixSimilarity(common_corpus, num_features=len(common_dictionary))
        >>> sims = index[query]

    """

    def __init__(self, corpus, num_best=None, dtype=numpy.float32, num_features=None, chunksize=256, corpus_len=None):
        """

        Parameters
        ----------
        corpus : iterable of list of (int, number)
            Corpus in streamed Gensim bag-of-words format.
        num_best : int, optional
            If set, return only the `num_best` most similar documents, always leaving out documents with similarity = 0.
            Otherwise, return a full vector with one float for every document in the index.
        num_features : int
            Size of the dictionary (number of features).
        corpus_len : int, optional
            Number of documents in `corpus`. If not specified, will scan the corpus to determine the matrix size.
        chunksize : int, optional
            Size of query chunks. Used internally when the query is an entire corpus.
        dtype : numpy.dtype, optional
            Datatype to store the internal matrix in.

        """
        if num_features is None:
            logger.warning('scanning corpus to determine the number of features (consider setting `num_features` explicitly)')
            num_features = 1 + utils.get_max_id(corpus)
        self.num_features = num_features
        self.num_best = num_best
        self.normalize = True
        self.chunksize = chunksize
        if corpus_len is None:
            corpus_len = len(corpus)
        if corpus is not None:
            if self.num_features <= 0:
                raise ValueError('cannot index a corpus with zero features (you must specify either `num_features` or a non-empty corpus in the constructor)')
            logger.info('creating matrix with %i documents and %i features', corpus_len, num_features)
            self.index = numpy.empty(shape=(corpus_len, num_features), dtype=dtype)
            for docno, vector in enumerate(corpus):
                if docno % 1000 == 0:
                    logger.debug('PROGRESS: at document #%i/%i', docno, corpus_len)
                if isinstance(vector, numpy.ndarray):
                    pass
                elif scipy.sparse.issparse(vector):
                    vector = vector.toarray().flatten()
                else:
                    vector = matutils.unitvec(matutils.sparse2full(vector, num_features))
                self.index[docno] = vector

    def __len__(self):
        return self.index.shape[0]

    def get_similarities(self, query):
        """Get similarity between `query` and this index.

        Warnings
        --------
        Do not use this function directly, use the :class:`~gensim.similarities.docsim.MatrixSimilarity.__getitem__`
        instead.

        Parameters
        ----------
        query : {list of (int, number), iterable of list of (int, number), :class:`scipy.sparse.csr_matrix`}
            Document or collection of documents.

        Return
        ------
        :class:`numpy.ndarray`
            Similarity matrix.

        """
        is_corpus, query = utils.is_corpus(query)
        if is_corpus:
            query = numpy.asarray([matutils.sparse2full(vec, self.num_features) for vec in query], dtype=self.index.dtype)
        else:
            if scipy.sparse.issparse(query):
                query = query.toarray()
            elif isinstance(query, numpy.ndarray):
                pass
            else:
                query = matutils.sparse2full(query, self.num_features)
            query = numpy.asarray(query, dtype=self.index.dtype)
        result = numpy.dot(self.index, query.T).T
        return result

    def __str__(self):
        return '%s<%i docs, %i features>' % (self.__class__.__name__, len(self), self.index.shape[1])