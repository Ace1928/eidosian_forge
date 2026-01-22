import itertools
import logging
import numpy as np
import scipy.sparse as sps
from gensim.topic_coherence.direct_confirmation_measure import aggregate_segment_sims, log_ratio_measure
def _map_to_contiguous(ids_iterable):
    uniq_ids = {}
    n = 0
    for id_ in itertools.chain.from_iterable(ids_iterable):
        if id_ not in uniq_ids:
            uniq_ids[id_] = n
            n += 1
    return uniq_ids