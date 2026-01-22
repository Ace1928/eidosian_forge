import itertools
import logging
import numpy as np
import scipy.sparse as sps
from gensim.topic_coherence.direct_confirmation_measure import aggregate_segment_sims, log_ratio_measure
class ContextVectorComputer:
    """Lazily compute context vectors for topic segments.

    Parameters
    ----------
    measure: str
        Confirmation measure.
    topics: list of numpy.array
        Topics.
    accumulator : :class:`~gensim.topic_coherence.text_analysis.WordVectorsAccumulator` or
                  :class:`~gensim.topic_coherence.text_analysis.InvertedIndexAccumulator`
        Word occurrence accumulator from probability_estimation.
    gamma: float
        Value for computing vectors.

    Attributes
    ----------
    sim_cache: dict
        Cache similarities between tokens (pairs of word ids), e.g. (1, 2).
    context_vector_cache: dict
        Mapping from (segment, topic_words) --> context_vector.

    Example
    -------
    .. sourcecode:: pycon

        >>> from gensim.corpora.dictionary import Dictionary
        >>> from gensim.topic_coherence import indirect_confirmation_measure, text_analysis
        >>> import numpy as np
        >>>
        >>> # create measure, topics
        >>> measure = 'nlr'
        >>> topics = [np.array([1, 2])]
        >>>
        >>> # create accumulator
        >>> dictionary = Dictionary()
        >>> dictionary.id2token = {1: 'fake', 2: 'tokens'}
        >>> accumulator = text_analysis.WordVectorsAccumulator({1, 2}, dictionary)
        >>> _ = accumulator.accumulate([['fake', 'tokens'], ['tokens', 'fake']], 5)
        >>> cont_vect_comp = indirect_confirmation_measure.ContextVectorComputer(measure, topics, accumulator, 1)
        >>> cont_vect_comp.mapping
        {1: 0, 2: 1}
        >>> cont_vect_comp.vocab_size
        2

    """

    def __init__(self, measure, topics, accumulator, gamma):
        if measure == 'nlr':
            self.similarity = _pair_npmi
        else:
            raise ValueError('The direct confirmation measure you entered is not currently supported.')
        self.mapping = _map_to_contiguous(topics)
        self.vocab_size = len(self.mapping)
        self.accumulator = accumulator
        self.gamma = gamma
        self.sim_cache = {}
        self.context_vector_cache = {}

    def __getitem__(self, idx):
        return self.compute_context_vector(*idx)

    def compute_context_vector(self, segment_word_ids, topic_word_ids):
        """Check if (segment_word_ids, topic_word_ids) context vector has been cached.

        Parameters
        ----------
        segment_word_ids: list
            Ids of words in segment.
        topic_word_ids: list
            Ids of words in topic.
        Returns
        -------
        csr_matrix :class:`~scipy.sparse.csr`
            If context vector has been cached, then return corresponding context vector,
            else compute, cache, and return.

        """
        key = _key_for_segment(segment_word_ids, topic_word_ids)
        context_vector = self.context_vector_cache.get(key, None)
        if context_vector is None:
            context_vector = self._make_seg(segment_word_ids, topic_word_ids)
            self.context_vector_cache[key] = context_vector
        return context_vector

    def _make_seg(self, segment_word_ids, topic_word_ids):
        """Return context vectors for segmentation (Internal helper function).

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

        """
        context_vector = sps.lil_matrix((self.vocab_size, 1))
        if not hasattr(segment_word_ids, '__iter__'):
            segment_word_ids = (segment_word_ids,)
        for w_j in topic_word_ids:
            idx = (self.mapping[w_j], 0)
            for pair in (tuple(sorted((w_i, w_j))) for w_i in segment_word_ids):
                if pair not in self.sim_cache:
                    self.sim_cache[pair] = self.similarity(pair, self.accumulator)
                context_vector[idx] += self.sim_cache[pair] ** self.gamma
        return context_vector.tocsr()