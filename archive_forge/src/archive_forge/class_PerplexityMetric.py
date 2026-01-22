import gensim
import logging
import copy
import sys
import numpy as np
class PerplexityMetric(Metric):
    """Metric class for perplexity evaluation."""

    def __init__(self, corpus=None, logger=None, viz_env=None, title=None):
        """

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional
            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).
        logger : {'shell', 'visdom'}, optional
           Monitor training process using one of the available methods. 'shell' will print the perplexity value in
           the active shell, while 'visdom' will visualize the coherence value with increasing epochs using the Visdom
           visualization framework.
        viz_env : object, optional
            Visdom environment to use for plotting the graph. Unused.
        title : str, optional
            Title of the graph plot in case `logger == 'visdom'`. Unused.

        """
        self.corpus = corpus
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

    def get_value(self, **kwargs):
        """Get the coherence score.

        Parameters
        ----------
        **kwargs
            Key word arguments to override the object's internal attributes.
            A trained topic model is expected using the 'model' key.
            This must be of type :class:`~gensim.models.ldamodel.LdaModel`.

        Returns
        -------
        float
            The perplexity score.

        """
        super(PerplexityMetric, self).set_parameters(**kwargs)
        corpus_words = sum((cnt for document in self.corpus for _, cnt in document))
        perwordbound = self.model.bound(self.corpus) / corpus_words
        return np.exp2(-perwordbound)