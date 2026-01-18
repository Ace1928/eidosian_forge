import unittest
from collections import defaultdict
from nltk.translate import AlignedSent, IBMModel, IBMModel4, IBMModel5
from nltk.translate.ibm_model import AlignmentInfo
def test_set_uniform_vacancy_probabilities_of_non_domain_values(self):
    src_classes = {'schinken': 0, 'eier': 0, 'spam': 1}
    trg_classes = {'ham': 0, 'eggs': 1, 'spam': 2}
    corpus = [AlignedSent(['ham', 'eggs'], ['schinken', 'schinken', 'eier']), AlignedSent(['spam', 'spam', 'spam', 'spam'], ['spam', 'spam'])]
    model5 = IBMModel5(corpus, 0, src_classes, trg_classes)
    model5.set_uniform_probabilities(corpus)
    self.assertEqual(model5.head_vacancy_table[5][4][0], IBMModel.MIN_PROB)
    self.assertEqual(model5.head_vacancy_table[-4][1][2], IBMModel.MIN_PROB)
    self.assertEqual(model5.head_vacancy_table[4][0][0], IBMModel.MIN_PROB)
    self.assertEqual(model5.non_head_vacancy_table[5][4][0], IBMModel.MIN_PROB)
    self.assertEqual(model5.non_head_vacancy_table[-4][1][2], IBMModel.MIN_PROB)