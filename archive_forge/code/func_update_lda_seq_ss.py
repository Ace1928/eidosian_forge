import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def update_lda_seq_ss(self, time, doc, topic_suffstats):
    """Update lda sequence sufficient statistics from an lda posterior.

        This is very similar to the :meth:`~gensim.models.ldaseqmodel.LdaPost.update_gamma` method and uses
        the same formula.

        Parameters
        ----------
        time : int
            The time slice.
        doc : list of (int, float)
            Unused but kept here for backwards compatibility. The document set in the constructor (`self.doc`) is used
            instead.
        topic_suffstats : list of float
            Sufficient statistics for each topic.

        Returns
        -------
        list of float
            The updated sufficient statistics for each topic.

        """
    num_topics = self.lda.num_topics
    for k in range(num_topics):
        topic_ss = topic_suffstats[k]
        n = 0
        for word_id, count in self.doc:
            topic_ss[word_id][time] += count * self.phi[n][k]
            n += 1
        topic_suffstats[k] = topic_ss
    return topic_suffstats