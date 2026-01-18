import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_tokenize_examples(self):
    """
        Test tokenization of the TGrep2 manual example patterns.
        """
    self.assertEqual(tgrep.tgrep_tokenize('NP < PP'), ['NP', '<', 'PP'])
    self.assertEqual(tgrep.tgrep_tokenize('/^NP/'), ['/^NP/'])
    self.assertEqual(tgrep.tgrep_tokenize('NP << PP . VP'), ['NP', '<<', 'PP', '.', 'VP'])
    self.assertEqual(tgrep.tgrep_tokenize('NP << PP | . VP'), ['NP', '<<', 'PP', '|', '.', 'VP'])
    self.assertEqual(tgrep.tgrep_tokenize('NP !<< PP [> NP | >> VP]'), ['NP', '!', '<<', 'PP', '[', '>', 'NP', '|', '>>', 'VP', ']'])
    self.assertEqual(tgrep.tgrep_tokenize('NP << (PP . VP)'), ['NP', '<<', '(', 'PP', '.', 'VP', ')'])
    self.assertEqual(tgrep.tgrep_tokenize("NP <' (PP <, (IN < on))"), ['NP', "<'", '(', 'PP', '<,', '(', 'IN', '<', 'on', ')', ')'])
    self.assertEqual(tgrep.tgrep_tokenize('S < (A < B) < C'), ['S', '<', '(', 'A', '<', 'B', ')', '<', 'C'])
    self.assertEqual(tgrep.tgrep_tokenize('S < ((A < B) < C)'), ['S', '<', '(', '(', 'A', '<', 'B', ')', '<', 'C', ')'])
    self.assertEqual(tgrep.tgrep_tokenize('S < (A < B < C)'), ['S', '<', '(', 'A', '<', 'B', '<', 'C', ')'])
    self.assertEqual(tgrep.tgrep_tokenize('A<B&.C'), ['A', '<', 'B', '&', '.', 'C'])