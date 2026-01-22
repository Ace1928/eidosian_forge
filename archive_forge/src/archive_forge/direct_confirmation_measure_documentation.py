import logging
import numpy as np
Compute log ratio measure for `segment_topics`.

    Parameters
    ----------
    segmented_topics : list of lists of (int, int)
        Output from the :func:`~gensim.topic_coherence.segmentation.s_one_pre`,
        :func:`~gensim.topic_coherence.segmentation.s_one_one`.
    accumulator : :class:`~gensim.topic_coherence.text_analysis.InvertedIndexAccumulator`
        Word occurrence accumulator from :mod:`gensim.topic_coherence.probability_estimation`.
    normalize : bool, optional
        Details in the "Notes" section.
    with_std : bool, optional
        True to also include standard deviation across topic segment sets in addition to the mean coherence
        for each topic.
    with_support : bool, optional
        True to also include support across topic segments. The support is defined as the number of pairwise
        similarity comparisons were used to compute the overall topic coherence.

    Notes
    -----
    If `normalize=False`:
        Calculate the log-ratio-measure, popularly known as **PMI** which is used by coherence measures such as `c_v`.
        This is defined as :math:`m_{lr}(S_i) = log \frac{P(W', W^{*}) + \epsilon}{P(W') * P(W^{*})}`

    If `normalize=True`:
        Calculate the normalized-log-ratio-measure, popularly knowns as **NPMI**
        which is used by coherence measures such as `c_v`.
        This is defined as :math:`m_{nlr}(S_i) = \frac{m_{lr}(S_i)}{-log(P(W', W^{*}) + \epsilon)}`

    Returns
    -------
    list of float
        Log ratio measurements for each topic.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.topic_coherence import direct_confirmation_measure, text_analysis
        >>> from collections import namedtuple
        >>>
        >>> # Create dictionary
        >>> id2token = {1: 'test', 2: 'doc'}
        >>> token2id = {v: k for k, v in id2token.items()}
        >>> dictionary = namedtuple('Dictionary', 'token2id, id2token')(token2id, id2token)
        >>>
        >>> # Initialize segmented topics and accumulator
        >>> segmentation = [[(1, 2)]]
        >>>
        >>> accumulator = text_analysis.InvertedIndexAccumulator({1, 2}, dictionary)
        >>> accumulator._inverted_index = {0: {2, 3, 4}, 1: {3, 5}}
        >>> accumulator._num_docs = 5
        >>>
        >>> # result should be ~ ln{(1 / 5) / [(3 / 5) * (2 / 5)]} = -0.182321557
        >>> result = direct_confirmation_measure.log_ratio_measure(segmentation, accumulator)[0]

    