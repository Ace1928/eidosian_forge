from the paper `Michael Roeder, Andreas Both and Alexander Hinneburg: "Exploring the space of topic coherence measures"
import logging
import multiprocessing as mp
from collections import namedtuple
import numpy as np
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
def _relevant_ids_will_differ(self, new_topics):
    if self._accumulator is None or not self._topics_differ(new_topics):
        return False
    new_set = unique_ids_from_segments(self.measure.seg(new_topics))
    return not self._accumulator.relevant_ids.issuperset(new_set)