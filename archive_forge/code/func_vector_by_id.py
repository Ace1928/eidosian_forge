import logging
import itertools
import os
import heapq
import warnings
import numpy
import scipy.sparse
from gensim import interfaces, utils, matutils
def vector_by_id(self, docpos):
    """Get the indexed vector corresponding to the document at position `docpos`.

        Parameters
        ----------
        docpos : int
            Document position

        Return
        ------
        :class:`scipy.sparse.csr_matrix`
            Indexed vector.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora.textcorpus import TextCorpus
            >>> from gensim.test.utils import datapath
            >>> from gensim.similarities import Similarity
            >>>
            >>> # Create index:
            >>> corpus = TextCorpus(datapath('testcorpus.txt'))
            >>> index = Similarity('temp', corpus, num_features=400)
            >>> vector = index.vector_by_id(1)

        """
    self.close_shard()
    pos = 0
    for shard in self.shards:
        pos += len(shard)
        if docpos < pos:
            break
    if not self.shards or docpos < 0 or docpos >= pos:
        raise ValueError('invalid document position: %s (must be 0 <= x < %s)' % (docpos, len(self)))
    result = shard.get_document_id(docpos - pos + len(shard))
    return result