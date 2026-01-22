from the paper `Michael Roeder, Andreas Both and Alexander Hinneburg: "Exploring the space of topic coherence measures"
import logging
import multiprocessing as mp
from collections import namedtuple
import numpy as np
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
def _ensure_elements_are_ids(self, topic):
    ids_from_tokens = [self.dictionary.token2id[t] for t in topic if t in self.dictionary.token2id]
    ids_from_ids = [i for i in topic if i in self.dictionary]
    if len(ids_from_tokens) > len(ids_from_ids):
        return np.array(ids_from_tokens)
    elif len(ids_from_ids) > len(ids_from_tokens):
        return np.array(ids_from_ids)
    else:
        raise ValueError('unable to interpret topic as either a list of tokens or a list of ids')