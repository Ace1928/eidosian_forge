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
def top_topics_as_word_lists(model, dictionary, topn=20):
    """Get `topn` topics as list of words.

        Parameters
        ----------
        model : :class:`~gensim.models.basemodel.BaseTopicModel`
            Pre-trained topic model.
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
            Gensim dictionary mapping of id word.
        topn : int, optional
            Integer corresponding to the number of top words to be extracted from each topic.

        Return
        ------
        list of list of str
            Top topics in list-of-list-of-words format.

        """
    if not dictionary.id2token:
        dictionary.id2token = {v: k for k, v in dictionary.token2id.items()}
    str_topics = []
    for topic in model.get_topics():
        bestn = matutils.argsort(topic, topn=topn, reverse=True)
        beststr = [dictionary.id2token[_id] for _id in bestn]
        str_topics.append(beststr)
    return str_topics