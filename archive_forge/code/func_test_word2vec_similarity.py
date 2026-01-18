import logging
import unittest
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.topic_coherence import indirect_confirmation_measure
from gensim.topic_coherence import text_analysis
def test_word2vec_similarity(self):
    """Sanity check word2vec_similarity."""
    accumulator = text_analysis.WordVectorsAccumulator({1, 2}, self.dictionary)
    accumulator.accumulate([['fake', 'tokens'], ['tokens', 'fake']], 5)
    mean, std = indirect_confirmation_measure.word2vec_similarity(self.segmentation, accumulator, with_std=True)[0]
    self.assertNotEqual(0.0, mean)
    self.assertNotEqual(0.0, std)