import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
class TestFrozenPhrasesPersistence(PhrasesData, unittest.TestCase):

    def test_save_load_custom_scorer(self):
        """Test saving and loading a FrozenPhrases object with a custom scorer."""
        with temporary_file('test.pkl') as fpath:
            bigram = FrozenPhrases(Phrases(self.sentences, min_count=1, threshold=0.001, scoring=dumb_scorer))
            bigram.save(fpath)
            bigram_loaded = FrozenPhrases.load(fpath)
            self.assertEqual(bigram_loaded.scoring, dumb_scorer)

    def test_save_load(self):
        """Test saving and loading a FrozenPhrases object."""
        with temporary_file('test.pkl') as fpath:
            bigram = FrozenPhrases(Phrases(self.sentences, min_count=1, threshold=1))
            bigram.save(fpath)
            bigram_loaded = FrozenPhrases.load(fpath)
            self.assertEqual(bigram_loaded[['graph', 'minors', 'survey', 'human', 'interface', 'system']], ['graph_minors', 'survey', 'human_interface', 'system'])

    def test_save_load_with_connector_words(self):
        """Test saving and loading a FrozenPhrases object."""
        connector_words = frozenset({'of'})
        with temporary_file('test.pkl') as fpath:
            bigram = FrozenPhrases(Phrases(self.sentences, min_count=1, threshold=1, connector_words=connector_words))
            bigram.save(fpath)
            bigram_loaded = FrozenPhrases.load(fpath)
            self.assertEqual(bigram_loaded.connector_words, connector_words)

    def test_save_load_string_scoring(self):
        """Test saving and loading a FrozenPhrases object with a string scoring parameter.
        This should ensure backwards compatibility with the previous version of FrozenPhrases"""
        bigram_loaded = FrozenPhrases.load(datapath('phraser-scoring-str.pkl'))
        self.assertEqual(bigram_loaded.scoring, original_scorer)

    def test_save_load_no_scoring(self):
        """Test saving and loading a FrozenPhrases object with no scoring parameter.
        This should ensure backwards compatibility with old versions of FrozenPhrases"""
        bigram_loaded = FrozenPhrases.load(datapath('phraser-no-scoring.pkl'))
        self.assertEqual(bigram_loaded.scoring, original_scorer)

    def test_save_load_no_common_terms(self):
        """Ensure backwards compatibility with old versions of FrozenPhrases, before connector_words."""
        bigram_loaded = FrozenPhrases.load(datapath('phraser-no-common-terms.pkl'))
        self.assertEqual(bigram_loaded.connector_words, frozenset())