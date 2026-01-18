import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test_pruning(self):
    """Test that max_vocab_size parameter is respected."""
    bigram = Phrases(self.sentences, max_vocab_size=5)
    self.assertTrue(len(bigram.vocab) <= 5)