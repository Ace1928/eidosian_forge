import logging
import itertools
from math import log
import pickle
from inspect import getfullargspec as getargspec
import time
from gensim import utils, interfaces


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

        