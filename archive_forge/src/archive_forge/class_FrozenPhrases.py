import logging
import itertools
from math import log
import pickle
from inspect import getfullargspec as getargspec
import time
from gensim import utils, interfaces
class FrozenPhrases(_PhrasesTransformation):
    """Minimal state & functionality exported from a trained :class:`~gensim.models.phrases.Phrases` model.

    The goal of this class is to cut down memory consumption of `Phrases`, by discarding model state
    not strictly needed for the phrase detection task.

    Use this instead of `Phrases` if you do not need to update the bigram statistics with new documents any more.

    """

    def __init__(self, phrases_model):
        """

        Parameters
        ----------
        phrases_model : :class:`~gensim.models.phrases.Phrases`
            Trained phrases instance, to extract all phrases from.

        Notes
        -----
        After the one-time initialization, a :class:`~gensim.models.phrases.FrozenPhrases` will be much
        smaller and faster than using the full :class:`~gensim.models.phrases.Phrases` model.

        Examples
        ----------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>> from gensim.models.word2vec import Text8Corpus
            >>> from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
            >>>
            >>> # Load corpus and train a model.
            >>> sentences = Text8Corpus(datapath('testcorpus.txt'))
            >>> phrases = Phrases(sentences, min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)
            >>>
            >>> # Export a FrozenPhrases object that is more efficient but doesn't allow further training.
            >>> frozen_phrases = phrases.freeze()
            >>> print(frozen_phrases[sent])
            [u'trees_graph', u'minors']

        """
        self.threshold = phrases_model.threshold
        self.min_count = phrases_model.min_count
        self.delimiter = phrases_model.delimiter
        self.scoring = phrases_model.scoring
        self.connector_words = phrases_model.connector_words
        logger.info('exporting phrases from %s', phrases_model)
        start = time.time()
        self.phrasegrams = phrases_model.export_phrases()
        self.add_lifecycle_event('created', msg=f'exported {self} from {phrases_model} in {time.time() - start:.2f}s')

    def __str__(self):
        return '%s<%i phrases, min_count=%s, threshold=%s>' % (self.__class__.__name__, len(self.phrasegrams), self.min_count, self.threshold)

    def score_candidate(self, word_a, word_b, in_between):
        phrase = self.delimiter.join([word_a] + in_between + [word_b])
        score = self.phrasegrams.get(phrase, NEGATIVE_INFINITY)
        if score > self.threshold:
            return (phrase, score)
        return (None, None)