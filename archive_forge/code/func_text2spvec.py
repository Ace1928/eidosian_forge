import numpy as np
import scipy.sparse as sp
from multiprocessing.pool import ThreadPool
from functools import partial
from . import utils
from . import tokenizers
from parlai.utils.logging import logger
def text2spvec(self, query):
    """
        Create a sparse tfidf-weighted word vector from query.

        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        """
    words = self.parse(utils.normalize(query))
    wids = [utils.hash(w, self.hash_size) for w in words]
    if len(wids) == 0:
        if self.strict:
            raise RuntimeError('No valid word in: %s' % query)
        else:
            logger.warning('No valid word in: %s' % query)
            return sp.csr_matrix((1, self.hash_size))
    wids_unique, wids_counts = np.unique(wids, return_counts=True)
    tfs = np.log1p(wids_counts)
    Ns = self.doc_freqs[wids_unique]
    idfs = np.log((self.num_docs - Ns + 0.5) / (Ns + 0.5))
    idfs[idfs < 0] = 0
    data = np.multiply(tfs, idfs)
    indptr = np.array([0, len(wids_unique)])
    spvec = sp.csr_matrix((data, wids_unique, indptr), shape=(1, self.hash_size))
    return spvec