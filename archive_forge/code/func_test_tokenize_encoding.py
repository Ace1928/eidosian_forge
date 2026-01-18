import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_tokenize_encoding(self):
    """
        Test that tokenization handles bytes and strs the same way.
        """
    self.assertEqual(tgrep.tgrep_tokenize(b'A .. (B !< C . D) | ![<< (E , F) $ G]'), tgrep.tgrep_tokenize('A .. (B !< C . D) | ![<< (E , F) $ G]'))