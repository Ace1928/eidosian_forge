from `Wang, Paisley, Blei: "Online Variational Inference for the Hierarchical Dirichlet Process",  JMLR (2011)
import logging
import time
import warnings
import numpy as np
from scipy.special import gammaln, psi  # gamma function utils
from gensim import interfaces, utils, matutils
from gensim.matutils import dirichlet_expectation, mean_absolute_difference
from gensim.models import basemodel, ldamodel
from gensim.utils import deprecated
class HdpTopicFormatter:
    """Helper class for :class:`gensim.models.hdpmodel.HdpModel` to format the output of topics."""
    STYLE_GENSIM, STYLE_PRETTY = (1, 2)

    def __init__(self, dictionary=None, topic_data=None, topic_file=None, style=None):
        """Initialise the :class:`gensim.models.hdpmodel.HdpTopicFormatter` and store topic data in sorted order.

        Parameters
        ----------
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`,optional
            Dictionary for the input corpus.
        topic_data : numpy.ndarray, optional
            The term topic matrix.
        topic_file : {file-like object, str, pathlib.Path}
            File, filename, or generator to read. If the filename extension is .gz or .bz2, the file is first
            decompressed. Note that generators should return byte strings for Python 3k.
        style : bool, optional
            If True - get the topics as a list of strings, otherwise - get the topics as lists of (word, weight) pairs.

        Raises
        ------
        ValueError
            Either dictionary is None or both `topic_data` and `topic_file` is None.

        """
        if dictionary is None:
            raise ValueError('no dictionary!')
        if topic_data is not None:
            topics = topic_data
        elif topic_file is not None:
            topics = np.loadtxt('%s' % topic_file)
        else:
            raise ValueError('no topic data!')
        topics_sums = np.sum(topics, axis=1)
        idx = matutils.argsort(topics_sums, reverse=True)
        self.data = topics[idx]
        self.dictionary = dictionary
        if style is None:
            style = self.STYLE_GENSIM
        self.style = style

    def print_topics(self, num_topics=10, num_words=10):
        """Give the most probable `num_words` words from `num_topics` topics.
        Alias for :meth:`~gensim.models.hdpmodel.HdpTopicFormatter.show_topics`.

        Parameters
        ----------
        num_topics : int, optional
            Top `num_topics` to be printed.
        num_words : int, optional
            Top `num_words` most probable words to be printed from each topic.

        Returns
        -------
        list of (str, numpy.float) **or** list of str
            Output format for `num_words` words from `num_topics` topics depends on the value of `self.style` attribute.

        """
        return self.show_topics(num_topics, num_words, True)

    def show_topics(self, num_topics=10, num_words=10, log=False, formatted=True):
        """Give the most probable `num_words` words from `num_topics` topics.

        Parameters
        ----------
        num_topics : int, optional
            Top `num_topics` to be printed.
        num_words : int, optional
            Top `num_words` most probable words to be printed from each topic.
        log : bool, optional
            If True - log a message with level INFO on the logger object.
        formatted : bool, optional
            If True - get the topics as a list of strings, otherwise as lists of (word, weight) pairs.

        Returns
        -------
        list of (int, list of (str, numpy.float) **or** list of str)
            Output format for terms from `num_topics` topics depends on the value of `self.style` attribute.

        """
        shown = []
        num_topics = max(num_topics, 0)
        num_topics = min(num_topics, len(self.data))
        for k in range(num_topics):
            lambdak = self.data[k, :]
            lambdak = lambdak / lambdak.sum()
            temp = zip(lambdak, range(len(lambdak)))
            temp = sorted(temp, key=lambda x: x[0], reverse=True)
            topic_terms = self.show_topic_terms(temp, num_words)
            if formatted:
                topic = self.format_topic(k, topic_terms)
                if log:
                    logger.info(topic)
            else:
                topic = (k, topic_terms)
            shown.append(topic)
        return shown

    def print_topic(self, topic_id, topn=None, num_words=None):
        """Print the `topn` most probable words from topic id `topic_id`.

        Warnings
        --------
        The parameter `num_words` is deprecated, will be removed in 4.0.0, please use `topn` instead.

        Parameters
        ----------
        topic_id : int
            Acts as a representative index for a particular topic.
        topn : int, optional
            Number of most probable words to show from given `topic_id`.
        num_words : int, optional
            DEPRECATED, USE `topn` INSTEAD.

        Returns
        -------
        list of (str, numpy.float) **or** list of str
            Output format for terms from a single topic depends on the value of `formatted` parameter.

        """
        if num_words is not None:
            warnings.warn('The parameter `num_words` is deprecated, will be removed in 4.0.0, please use `topn` instead.')
            topn = num_words
        return self.show_topic(topic_id, topn, formatted=True)

    def show_topic(self, topic_id, topn=20, log=False, formatted=False, num_words=None):
        """Give the most probable `num_words` words for the id `topic_id`.

        Warnings
        --------
        The parameter `num_words` is deprecated, will be removed in 4.0.0, please use `topn` instead.

        Parameters
        ----------
        topic_id : int
            Acts as a representative index for a particular topic.
        topn : int, optional
            Number of most probable words to show from given `topic_id`.
        log : bool, optional
            If True logs a message with level INFO on the logger object, False otherwise.
        formatted : bool, optional
            If True return the topics as a list of strings, False as lists of
            (word, weight) pairs.
        num_words : int, optional
            DEPRECATED, USE `topn` INSTEAD.

        Returns
        -------
        list of (str, numpy.float) **or** list of str
            Output format for terms from a single topic depends on the value of `self.style` attribute.

        """
        if num_words is not None:
            warnings.warn('The parameter `num_words` is deprecated, will be removed in 4.0.0, please use `topn` instead.')
            topn = num_words
        lambdak = self.data[topic_id, :]
        lambdak = lambdak / lambdak.sum()
        temp = zip(lambdak, range(len(lambdak)))
        temp = sorted(temp, key=lambda x: x[0], reverse=True)
        topic_terms = self.show_topic_terms(temp, topn)
        if formatted:
            topic = self.format_topic(topic_id, topic_terms)
            if log:
                logger.info(topic)
        else:
            topic = (topic_id, topic_terms)
        return topic[1]

    def show_topic_terms(self, topic_data, num_words):
        """Give the topic terms along with their probabilities for a single topic data.

        Parameters
        ----------
        topic_data : list of (str, numpy.float)
            Contains probabilities for each word id belonging to a single topic.
        num_words : int
            Number of words for which probabilities are to be extracted from the given single topic data.

        Returns
        -------
        list of (str, numpy.float)
            A sequence of topic terms and their probabilities.

        """
        return [(self.dictionary[wid], weight) for weight, wid in topic_data[:num_words]]

    def format_topic(self, topic_id, topic_terms):
        """Format the display for a single topic in two different ways.

        Parameters
        ----------
        topic_id : int
            Acts as a representative index for a particular topic.
        topic_terms : list of (str, numpy.float)
            Contains the most probable words from a single topic.

        Returns
        -------
        list of (str, numpy.float) **or** list of str
            Output format for topic terms depends on the value of `self.style` attribute.

        """
        if self.STYLE_GENSIM == self.style:
            fmt = ' + '.join(('%.3f*%s' % (weight, word) for word, weight in topic_terms))
        else:
            fmt = '\n'.join(('    %20s    %.8f' % (word, weight) for word, weight in topic_terms))
        fmt = (topic_id, fmt)
        return fmt