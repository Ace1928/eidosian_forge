import logging
import itertools
from math import log
import pickle
from inspect import getfullargspec as getargspec
import time
from gensim import utils, interfaces
class _PhrasesTransformation(interfaces.TransformationABC):
    """
    Abstract base class for :class:`~gensim.models.phrases.Phrases` and
    :class:`~gensim.models.phrases.FrozenPhrases`.

    """

    def __init__(self, connector_words):
        self.connector_words = frozenset(connector_words)

    def score_candidate(self, word_a, word_b, in_between):
        """Score a single phrase candidate.

        Returns
        -------
        (str, float)
            2-tuple of ``(delimiter-joined phrase, phrase score)`` for a phrase,
            or ``(None, None)`` if not a phrase.
        """
        raise NotImplementedError('ABC: override this method in child classes')

    def analyze_sentence(self, sentence):
        """Analyze a sentence, concatenating any detected phrases into a single token.

        Parameters
        ----------
        sentence : iterable of str
            Token sequence representing the sentence to be analyzed.

        Yields
        ------
        (str, {float, None})
            Iterate through the input sentence tokens and yield 2-tuples of:
            - ``(concatenated_phrase_tokens, score)`` for token sequences that form a phrase.
            - ``(word, None)`` if the token is not a part of a phrase.

        """
        start_token, in_between = (None, [])
        for word in sentence:
            if word not in self.connector_words:
                if start_token:
                    phrase, score = self.score_candidate(start_token, word, in_between)
                    if score is not None:
                        yield (phrase, score)
                        start_token, in_between = (None, [])
                    else:
                        yield (start_token, None)
                        for w in in_between:
                            yield (w, None)
                        start_token, in_between = (word, [])
                else:
                    start_token, in_between = (word, [])
            elif start_token:
                in_between.append(word)
            else:
                yield (word, None)
        if start_token:
            yield (start_token, None)
            for w in in_between:
                yield (w, None)

    def __getitem__(self, sentence):
        """Convert the input sequence of tokens ``sentence`` into a sequence of tokens where adjacent
        tokens are replaced by a single token if they form a bigram collocation.

        If `sentence` is an entire corpus (iterable of sentences rather than a single
        sentence), return an iterable that converts each of the corpus' sentences
        into phrases on the fly, one after another.

        Parameters
        ----------
        sentence : {list of str, iterable of list of str}
            Input sentence or a stream of sentences.

        Return
        ------
        {list of str, iterable of list of str}
            Sentence with phrase tokens joined by ``self.delimiter``, if input was a single sentence.
            A generator of such sentences if input was a corpus.

s        """
        is_single, sentence = _is_single(sentence)
        if not is_single:
            return self._apply(sentence)
        return [token for token, _ in self.analyze_sentence(sentence)]

    def find_phrases(self, sentences):
        """Get all unique phrases (multi-word expressions) that appear in ``sentences``, and their scores.

        Parameters
        ----------
        sentences : iterable of list of str
            Text corpus.

        Returns
        -------
        dict(str, float)
           Unique phrases found in ``sentences``, mapped to their scores.

        Example
        -------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>> from gensim.models.word2vec import Text8Corpus
            >>> from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
            >>>
            >>> sentences = Text8Corpus(datapath('testcorpus.txt'))
            >>> phrases = Phrases(sentences, min_count=1, threshold=0.1, connector_words=ENGLISH_CONNECTOR_WORDS)
            >>>
            >>> for phrase, score in phrases.find_phrases(sentences).items():
            ...     print(phrase, score)
        """
        result = {}
        for sentence in sentences:
            for phrase, score in self.analyze_sentence(sentence):
                if score is not None:
                    result[phrase] = score
        return result

    @classmethod
    def load(cls, *args, **kwargs):
        """Load a previously saved :class:`~gensim.models.phrases.Phrases` /
        :class:`~gensim.models.phrases.FrozenPhrases` model.

        Handles backwards compatibility from older versions which did not support pluggable scoring functions.

        Parameters
        ----------
        args : object
            See :class:`~gensim.utils.SaveLoad.load`.
        kwargs : object
            See :class:`~gensim.utils.SaveLoad.load`.

        """
        model = super(_PhrasesTransformation, cls).load(*args, **kwargs)
        try:
            phrasegrams = getattr(model, 'phrasegrams', {})
            component, score = next(iter(phrasegrams.items()))
            if isinstance(score, tuple):
                model.phrasegrams = {str(model.delimiter.join(key), encoding='utf8'): val[1] for key, val in phrasegrams.items()}
            elif isinstance(component, tuple):
                model.phrasegrams = {str(model.delimiter.join(key), encoding='utf8'): val for key, val in phrasegrams.items()}
        except StopIteration:
            pass
        if not hasattr(model, 'scoring'):
            logger.warning('older version of %s loaded without scoring function', cls.__name__)
            logger.warning('setting pluggable scoring method to original_scorer for compatibility')
            model.scoring = original_scorer
        if hasattr(model, 'scoring'):
            if isinstance(model.scoring, str):
                if model.scoring == 'default':
                    logger.warning('older version of %s loaded with "default" scoring parameter', cls.__name__)
                    logger.warning('setting scoring method to original_scorer for compatibility')
                    model.scoring = original_scorer
                elif model.scoring == 'npmi':
                    logger.warning('older version of %s loaded with "npmi" scoring parameter', cls.__name__)
                    logger.warning('setting scoring method to npmi_scorer for compatibility')
                    model.scoring = npmi_scorer
                else:
                    raise ValueError(f'failed to load {cls.__name__} model, unknown scoring "{model.scoring}"')
        if not hasattr(model, 'connector_words'):
            if hasattr(model, 'common_terms'):
                model.connector_words = model.common_terms
                del model.common_terms
            else:
                logger.warning('loaded older version of %s, setting connector_words to an empty set', cls.__name__)
                model.connector_words = frozenset()
        if not hasattr(model, 'corpus_word_count'):
            logger.warning('older version of %s loaded without corpus_word_count', cls.__name__)
            logger.warning('setting corpus_word_count to 0, do not use it in your scoring function')
            model.corpus_word_count = 0
        if getattr(model, 'vocab', None):
            word = next(iter(model.vocab))
            if not isinstance(word, str):
                logger.info('old version of %s loaded, upgrading %i words in memory', cls.__name__, len(model.vocab))
                logger.info('re-save the loaded model to avoid this upgrade in the future')
                vocab = {}
                for key, value in model.vocab.items():
                    vocab[str(key, encoding='utf8')] = value
                model.vocab = vocab
        if not isinstance(model.delimiter, str):
            model.delimiter = str(model.delimiter, encoding='utf8')
        return model