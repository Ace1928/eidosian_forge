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
def write_vector(self, docno, vector):
    """Write a single sparse vector to the file.

        Parameters
        ----------
        docno : int
            Number of document.
        vector : list of (int, number)
            Document in BoW format.

        Returns
        -------
        (int, int)
            Max word index in vector and len of vector. If vector is empty, return (-1, 0).

        """
    assert self.headers_written, 'must write Matrix Market file headers before writing data!'
    assert self.last_docno < docno, 'documents %i and %i not in sequential order!' % (self.last_docno, docno)
    vector = sorted(((i, w) for i, w in vector if abs(w) > 1e-12))
    for termid, weight in vector:
        self.fout.write(utils.to_utf8('%i %i %s\n' % (docno + 1, termid + 1, weight)))
    self.last_docno = docno
    return (vector[-1][0], len(vector)) if vector else (-1, 0)