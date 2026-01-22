from __future__ import with_statement
import logging
import math
from gensim import utils
import numpy as np
import scipy.sparse
from scipy.stats import entropy
from scipy.linalg import get_blas_funcs, triu
from scipy.linalg.lapack import get_lapack_funcs
from scipy.special import psi  # gamma function utils
class Dense2Corpus:
    """Treat dense numpy array as a streamed Gensim corpus in the bag-of-words format.

    Notes
    -----
    No data copy is made (changes to the underlying matrix imply changes in the streamed corpus).

    See Also
    --------
    :func:`~gensim.matutils.corpus2dense`
        Convert Gensim corpus to dense matrix.
    :class:`~gensim.matutils.Sparse2Corpus`
        Convert sparse matrix to Gensim corpus format.

    """

    def __init__(self, dense, documents_columns=True):
        """

        Parameters
        ----------
        dense : numpy.ndarray
            Corpus in dense format.
        documents_columns : bool, optional
            Documents in `dense` represented as columns, as opposed to rows?

        """
        if documents_columns:
            self.dense = dense.T
        else:
            self.dense = dense

    def __iter__(self):
        """Iterate over the corpus.

        Yields
        ------
        list of (int, float)
            Document in BoW format.

        """
        for doc in self.dense:
            yield full2sparse(doc.flat)

    def __len__(self):
        return len(self.dense)