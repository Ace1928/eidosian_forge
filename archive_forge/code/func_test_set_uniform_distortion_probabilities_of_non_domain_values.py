import unittest
from collections import defaultdict
from nltk.translate import AlignedSent, IBMModel, IBMModel3
from nltk.translate.ibm_model import AlignmentInfo
def test_set_uniform_distortion_probabilities_of_non_domain_values(self):
    corpus = [AlignedSent(['ham', 'eggs'], ['schinken', 'schinken', 'eier']), AlignedSent(['spam', 'spam', 'spam', 'spam'], ['spam', 'spam'])]
    model3 = IBMModel3(corpus, 0)
    model3.set_uniform_probabilities(corpus)
    self.assertEqual(model3.distortion_table[0][0][3][2], IBMModel.MIN_PROB)
    self.assertEqual(model3.distortion_table[9][2][2][4], IBMModel.MIN_PROB)
    self.assertEqual(model3.distortion_table[2][9][2][4], IBMModel.MIN_PROB)