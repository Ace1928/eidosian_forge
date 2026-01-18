import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_tokenize_quoting(self):
    """
        Test tokenization of quoting.
        """
    self.assertEqual(tgrep.tgrep_tokenize('"A<<:B"<<:"A $.. B"<"A>3B"<C'), ['"A<<:B"', '<<:', '"A $.. B"', '<', '"A>3B"', '<', 'C'])