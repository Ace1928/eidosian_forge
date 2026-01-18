import itertools
import logging
from gensim.topic_coherence.text_analysis import (
Return the set of all unique ids in a list of segmented topics.

    Parameters
    ----------
    segmented_topics: list of (int, int).
        Each tuple (word_id_set1, word_id_set2) is either a single integer, or a `numpy.ndarray` of integers.

    Returns
    -------
    set
        Set of unique ids across all topic segments.

    Example
    -------
    .. sourcecode:: pycon

        >>> from gensim.topic_coherence import probability_estimation
        >>>
        >>> segmentation = [[(1, 2)]]
        >>> probability_estimation.unique_ids_from_segments(segmentation)
        set([1, 2])

    