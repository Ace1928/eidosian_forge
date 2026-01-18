import logging
import numbers
import os
import time
from collections import defaultdict
import numpy as np
from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma
from gensim import interfaces, utils, matutils
from gensim.matutils import (
from gensim.models import basemodel, CoherenceModel
from gensim.models.callbacks import Callback
def top_topics(self, corpus=None, texts=None, dictionary=None, window_size=None, coherence='u_mass', topn=20, processes=-1):
    """Get the topics with the highest coherence score the coherence for each topic.

        Parameters
        ----------
        corpus : iterable of list of (int, float), optional
            Corpus in BoW format.
        texts : list of list of str, optional
            Tokenized texts, needed for coherence models that use sliding window based (i.e. coherence=`c_something`)
            probability estimator .
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            Gensim dictionary mapping of id word to create corpus.
            If `model.id2word` is present, this is not needed. If both are provided, passed `dictionary` will be used.
        window_size : int, optional
            Is the size of the window to be used for coherence measures using boolean sliding window as their
            probability estimator. For 'u_mass' this doesn't matter.
            If None - the default window sizes are used which are: 'c_v' - 110, 'c_uci' - 10, 'c_npmi' - 10.
        coherence : {'u_mass', 'c_v', 'c_uci', 'c_npmi'}, optional
            Coherence measure to be used.
            Fastest method - 'u_mass', 'c_uci' also known as `c_pmi`.
            For 'u_mass' corpus should be provided, if texts is provided, it will be converted to corpus
            using the dictionary. For 'c_v', 'c_uci' and 'c_npmi' `texts` should be provided (`corpus` isn't needed)
        topn : int, optional
            Integer corresponding to the number of top words to be extracted from each topic.
        processes : int, optional
            Number of processes to use for probability estimation phase, any value less than 1 will be interpreted as
            num_cpus - 1.

        Returns
        -------
        list of (list of (int, str), float)
            Each element in the list is a pair of a topic representation and its coherence score. Topic representations
            are distributions of words, represented as a list of pairs of word IDs and their probabilities.

        """
    cm = CoherenceModel(model=self, corpus=corpus, texts=texts, dictionary=dictionary, window_size=window_size, coherence=coherence, topn=topn, processes=processes)
    coherence_scores = cm.get_coherence_per_topic()
    str_topics = []
    for topic in self.get_topics():
        bestn = matutils.argsort(topic, topn=topn, reverse=True)
        beststr = [(topic[_id], self.id2word[_id]) for _id in bestn]
        str_topics.append(beststr)
    scored_topics = zip(str_topics, coherence_scores)
    return sorted(scored_topics, key=lambda tup: tup[1], reverse=True)