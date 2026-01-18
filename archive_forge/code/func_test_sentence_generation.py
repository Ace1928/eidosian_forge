import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test_sentence_generation(self):
    """Test basic bigram using a dummy corpus."""
    self.assertEqual(len(self.sentences), len(list(self.bigram_default[self.sentences])))