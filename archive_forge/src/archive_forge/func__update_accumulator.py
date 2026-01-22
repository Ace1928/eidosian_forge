from the paper `Michael Roeder, Andreas Both and Alexander Hinneburg: "Exploring the space of topic coherence measures"
import logging
import multiprocessing as mp
from collections import namedtuple
import numpy as np
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
def _update_accumulator(self, new_topics):
    if self._relevant_ids_will_differ(new_topics):
        logger.debug('Wiping cached accumulator since it does not contain all relevant ids.')
        self._accumulator = None