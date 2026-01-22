from the paper `Michael Roeder, Andreas Both and Alexander Hinneburg: "Exploring the space of topic coherence measures"
import logging
import multiprocessing as mp
from collections import namedtuple
import numpy as np
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
@staticmethod
def _get_topics_from_model(model, topn):
    """Internal helper function to return topics from a trained topic model.

        Parameters
        ----------
        model : :class:`~gensim.models.basemodel.BaseTopicModel`
            Pre-trained topic model.
        topn : int
            Integer corresponding to the number of top words.

        Return
        ------
        list of :class:`numpy.ndarray`
            Topics matrix

        """
    try:
        return [matutils.argsort(topic, topn=topn, reverse=True) for topic in model.get_topics()]
    except AttributeError:
        raise ValueError('This topic model is not currently supported. Supported topic models should implement the `get_topics` method.')