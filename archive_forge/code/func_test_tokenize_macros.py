import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_tokenize_macros(self):
    """
        Test tokenization of macro definitions.
        """
    self.assertEqual(tgrep.tgrep_tokenize('@ NP /^NP/;\n@ NN /^NN/;\n@NP [!< NP | < @NN] !$.. @NN'), ['@', 'NP', '/^NP/', ';', '@', 'NN', '/^NN/', ';', '@NP', '[', '!', '<', 'NP', '|', '<', '@NN', ']', '!', '$..', '@NN'])