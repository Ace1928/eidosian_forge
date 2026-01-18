import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_tokenize_simple(self):
    """
        Simple test of tokenization.
        """
    tokens = tgrep.tgrep_tokenize('A .. (B !< C . D) | ![<< (E , F) $ G]')
    self.assertEqual(tokens, ['A', '..', '(', 'B', '!', '<', 'C', '.', 'D', ')', '|', '!', '[', '<<', '(', 'E', ',', 'F', ')', '$', 'G', ']'])