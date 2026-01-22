import logging
import numpy as np
def aggregate_segment_sims(segment_sims, with_std, with_support):
    """Compute various statistics from the segment similarities generated via set pairwise comparisons
    of top-N word lists for a single topic.

    Parameters
    ----------
    segment_sims : iterable of float
        Similarity values to aggregate.
    with_std : bool
        Set to True to include standard deviation.
    with_support : bool
        Set to True to include number of elements in `segment_sims` as a statistic in the results returned.

    Returns
    -------
    (float[, float[, int]])
        Tuple with (mean[, std[, support]]).

    Examples
    ---------
    .. sourcecode:: pycon

        >>> from gensim.topic_coherence import direct_confirmation_measure
        >>>
        >>> segment_sims = [0.2, 0.5, 1., 0.05]
        >>> direct_confirmation_measure.aggregate_segment_sims(segment_sims, True, True)
        (0.4375, 0.36293077852394939, 4)
        >>> direct_confirmation_measure.aggregate_segment_sims(segment_sims, False, False)
        0.4375

    """
    mean = np.mean(segment_sims)
    stats = [mean]
    if with_std:
        stats.append(np.std(segment_sims))
    if with_support:
        stats.append(len(segment_sims))
    return stats[0] if len(stats) == 1 else tuple(stats)