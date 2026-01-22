from __future__ import print_function
import logging
import os
import math
import time
import numpy
import scipy.sparse as sparse
import gensim
from gensim.corpora import IndexedCorpus
from gensim.interfaces import TransformedCorpus
def __add_to_slice(self, s_result, result_start, result_stop, start, stop):
    """
        Add rows of the current shard from `start` to `stop`
        into rows `result_start` to `result_stop` of `s_result`.

        Operation is based on the ``self.sparse_serialize`` setting. If the shard
        contents are dense, then s_result is assumed to be an ndarray that
        already supports row indices `result_start:result_stop`. If the shard
        contents are sparse, assumes that s_result has `result_start` rows
        and we should add them up to `result_stop`.

        Return the resulting ``s_result``.

        """
    if result_stop - result_start != stop - start:
        raise ValueError('Result start/stop range different than stop/start range (%s - %s vs. %s - %s)' % (result_start, result_stop, start, stop))
    if not self.sparse_serialization:
        s_result[result_start:result_stop] = self.current_shard[start:stop]
        return s_result
    if s_result.shape != (result_start, self.dim):
        raise ValueError('Assuption about sparse s_result shape invalid: %s expected rows, %s real rows.' % (result_start, s_result.shape[0]))
    tmp_matrix = self.current_shard[start:stop]
    s_result = sparse.vstack([s_result, tmp_matrix])
    return s_result