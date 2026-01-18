import itertools
import logging
import numpy as np
import scipy.sparse as sps
from gensim.topic_coherence.direct_confirmation_measure import aggregate_segment_sims, log_ratio_measure
Return context vectors for segmentation (Internal helper function).

        Parameters
        ----------
        segment_word_ids : iterable or int
            Ids of words in segment.
        topic_word_ids : list
            Ids of words in topic.
        Returns
        -------
        csr_matrix :class:`~scipy.sparse.csr`
            Matrix in Compressed Sparse Row format

        