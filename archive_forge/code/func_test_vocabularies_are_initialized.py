import unittest
from collections import defaultdict
from nltk.translate import AlignedSent, IBMModel
from nltk.translate.ibm_model import AlignmentInfo
def test_vocabularies_are_initialized(self):
    parallel_corpora = [AlignedSent(['one', 'two', 'three', 'four'], ['un', 'deux', 'trois']), AlignedSent(['five', 'one', 'six'], ['quatre', 'cinq', 'six']), AlignedSent([], ['sept'])]
    ibm_model = IBMModel(parallel_corpora)
    self.assertEqual(len(ibm_model.src_vocab), 8)
    self.assertEqual(len(ibm_model.trg_vocab), 6)