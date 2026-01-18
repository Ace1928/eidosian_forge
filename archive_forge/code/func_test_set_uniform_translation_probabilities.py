import unittest
from collections import defaultdict
from nltk.translate import AlignedSent, IBMModel, IBMModel1
from nltk.translate.ibm_model import AlignmentInfo
def test_set_uniform_translation_probabilities(self):
    corpus = [AlignedSent(['ham', 'eggs'], ['schinken', 'schinken', 'eier']), AlignedSent(['spam', 'spam', 'spam', 'spam'], ['spam', 'spam'])]
    model1 = IBMModel1(corpus, 0)
    model1.set_uniform_probabilities(corpus)
    self.assertEqual(model1.translation_table['ham']['eier'], 1.0 / 3)
    self.assertEqual(model1.translation_table['eggs'][None], 1.0 / 3)