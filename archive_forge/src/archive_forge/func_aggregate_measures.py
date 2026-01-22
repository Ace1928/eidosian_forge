from the paper `Michael Roeder, Andreas Both and Alexander Hinneburg: "Exploring the space of topic coherence measures"
import logging
import multiprocessing as mp
from collections import namedtuple
import numpy as np
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
def aggregate_measures(self, topic_coherences):
    """Aggregate the individual topic coherence measures using the pipeline's aggregation function.
        Use `self.measure.aggr(topic_coherences)`.

        Parameters
        ----------
        topic_coherences : list of float
            List of calculated confirmation measure on each set in the segmented topics.

        Returns
        -------
        float
            Arithmetic mean of all the values contained in confirmation measures.

        """
    return self.measure.aggr(topic_coherences)