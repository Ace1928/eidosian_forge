from the paper `Michael Roeder, Andreas Both and Alexander Hinneburg: "Exploring the space of topic coherence measures"
import logging
import multiprocessing as mp
from collections import namedtuple
import numpy as np
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
@topn.setter
def topn(self, topn):
    """Set number of top words `self._topn`.

        Parameters
        ----------
        topn : int
            Number of top words.

        """
    current_topic_length = len(self._topics[0])
    requires_expansion = current_topic_length < topn
    if self.model is not None:
        self._topn = topn
        if requires_expansion:
            self.model = self._model
    else:
        if requires_expansion:
            raise ValueError('Model unavailable and topic sizes are less than topn=%d' % topn)
        self._topn = topn