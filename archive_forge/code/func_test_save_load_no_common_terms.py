import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test_save_load_no_common_terms(self):
    """Ensure backwards compatibility with old versions of FrozenPhrases, before connector_words."""
    bigram_loaded = FrozenPhrases.load(datapath('phraser-no-common-terms.pkl'))
    self.assertEqual(bigram_loaded.connector_words, frozenset())