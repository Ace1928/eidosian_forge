import logging
import itertools
import os
import heapq
import warnings
import numpy
import scipy.sparse
from gensim import interfaces, utils, matutils
class SparseMatrixSimilarity(interfaces.SimilarityABC):
    """Compute cosine similarity against a corpus of documents by storing the index matrix in memory.

    Examples
    --------
    Here is how you would index and query a corpus of documents in the bag-of-words format using the
    cosine similarity:

    .. sourcecode:: pycon

        >>> from gensim.corpora import Dictionary
        >>> from gensim.similarities import SparseMatrixSimilarity
        >>> from gensim.test.utils import common_texts as corpus
        >>>
        >>> dictionary = Dictionary(corpus)  # fit dictionary
        >>> bow_corpus = [dictionary.doc2bow(line) for line in corpus]  # convert corpus to BoW format
        >>> index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary))
        >>>
        >>> query = 'graph trees computer'.split()  # make a query
        >>> bow_query = dictionary.doc2bow(query)
        >>> similarities = index[bow_query]  # calculate similarity of query to each doc from bow_corpus

    Here is how you would index and query a corpus of documents using the Okapi BM25 scoring
    function:

    .. sourcecode:: pycon

        >>> from gensim.corpora import Dictionary
        >>> from gensim.models import TfidfModel, OkapiBM25Model
        >>> from gensim.similarities import SparseMatrixSimilarity
        >>> from gensim.test.utils import common_texts as corpus
        >>>
        >>> dictionary = Dictionary(corpus)  # fit dictionary
        >>> query_model = TfidfModel(dictionary=dictionary, smartirs='bnn')  # enforce binary weights
        >>> document_model = OkapiBM25Model(dictionary=dictionary)  # fit bm25 model
        >>>
        >>> bow_corpus = [dictionary.doc2bow(line) for line in corpus]  # convert corpus to BoW format
        >>> bm25_corpus = document_model[bow_corpus]
        >>> index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
        ...                                normalize_queries=False, normalize_documents=False)
        >>>
        >>>
        >>> query = 'graph trees computer'.split()  # make a query
        >>> bow_query = dictionary.doc2bow(query)
        >>> bm25_query = query_model[bow_query]
        >>> similarities = index[bm25_query]  # calculate similarity of query to each doc from bow_corpus

    Notes
    -----
    Use this if your input corpus contains sparse vectors (such as TF-IDF documents) and fits into RAM.

    The matrix is internally stored as a :class:`scipy.sparse.csr_matrix` matrix. Unless the entire
    matrix fits into main memory, use :class:`~gensim.similarities.docsim.Similarity` instead.

    Takes an optional `maintain_sparsity` argument, setting this to True
    causes `get_similarities` to return a sparse matrix instead of a
    dense representation if possible.

    See also
    --------
    :class:`~gensim.similarities.docsim.Similarity`
        Index similarity (wrapper for other inheritors of :class:`~gensim.interfaces.SimilarityABC`).
    :class:`~gensim.similarities.docsim.MatrixSimilarity`
        Index similarity (dense with cosine distance).

    """

    def __init__(self, corpus, num_features=None, num_terms=None, num_docs=None, num_nnz=None, num_best=None, chunksize=500, dtype=numpy.float32, maintain_sparsity=False, normalize_queries=True, normalize_documents=True):
        """

        Parameters
        ----------
        corpus: iterable of list of (int, float)
            A list of documents in the BoW format.
        num_features : int, optional
            Size of the dictionary. Must be either specified, or present in `corpus.num_terms`.
        num_terms : int, optional
            Alias for `num_features`, you can use either.
        num_docs : int, optional
            Number of documents in `corpus`. Will be calculated if not provided.
        num_nnz : int, optional
            Number of non-zero elements in `corpus`. Will be calculated if not provided.
        num_best : int, optional
            If set, return only the `num_best` most similar documents, always leaving out documents with similarity = 0.
            Otherwise, return a full vector with one float for every document in the index.
        chunksize : int, optional
            Size of query chunks. Used internally when the query is an entire corpus.
        dtype : numpy.dtype, optional
            Data type of the internal matrix.
        maintain_sparsity : bool, optional
            Return sparse arrays from :meth:`~gensim.similarities.docsim.SparseMatrixSimilarity.get_similarities`?
        normalize_queries : bool, optional
            If queries are in bag-of-words (int, float) format, as opposed to a sparse or dense
            2D arrays, they will be L2-normalized. Default is True.
        normalize_documents : bool, optional
            If `corpus` is in bag-of-words (int, float) format, as opposed to a sparse or dense
            2D arrays, it will be L2-normalized. Default is True.
        """
        self.num_best = num_best
        self.normalize = normalize_queries
        self.chunksize = chunksize
        self.maintain_sparsity = maintain_sparsity
        if corpus is not None:
            logger.info('creating sparse index')
            try:
                num_terms, num_docs, num_nnz = (corpus.num_terms, corpus.num_docs, corpus.num_nnz)
                logger.debug('using efficient sparse index creation')
            except AttributeError:
                pass
            if num_features is not None:
                num_terms = num_features
            if num_terms is None:
                raise ValueError('refusing to guess the number of sparse features: specify num_features explicitly')
            corpus = (matutils.scipy2sparse(v) if scipy.sparse.issparse(v) else matutils.full2sparse(v) if isinstance(v, numpy.ndarray) else matutils.unitvec(v) if normalize_documents else v for v in corpus)
            self.index = matutils.corpus2csc(corpus, num_terms=num_terms, num_docs=num_docs, num_nnz=num_nnz, dtype=dtype, printprogress=10000).T
            self.index = self.index.tocsr()
            logger.info('created %r', self.index)

    def __len__(self):
        """Get size of index."""
        return self.index.shape[0]

    def get_similarities(self, query):
        """Get similarity between `query` and this index.

        Warnings
        --------
        Do not use this function directly; use the `self[query]` syntax instead.

        Parameters
        ----------
        query : {list of (int, number), iterable of list of (int, number), :class:`scipy.sparse.csr_matrix`}
            Document or collection of documents.

        Return
        ------
        :class:`numpy.ndarray`
            Similarity matrix (if maintain_sparsity=False) **OR**
        :class:`scipy.sparse.csc`
            otherwise

        """
        is_corpus, query = utils.is_corpus(query)
        if is_corpus:
            query = matutils.corpus2csc(query, self.index.shape[1], dtype=self.index.dtype)
        elif scipy.sparse.issparse(query):
            query = query.T
        elif isinstance(query, numpy.ndarray):
            if query.ndim == 1:
                query.shape = (1, len(query))
            query = scipy.sparse.csr_matrix(query, dtype=self.index.dtype).T
        else:
            query = matutils.corpus2csc([query], self.index.shape[1], dtype=self.index.dtype)
        result = self.index * query.tocsc()
        if result.shape[1] == 1 and (not is_corpus):
            result = result.toarray().flatten()
        elif self.maintain_sparsity:
            result = result.T
        else:
            result = result.toarray().T
        return result