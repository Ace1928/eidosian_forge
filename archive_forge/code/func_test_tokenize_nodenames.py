import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_tokenize_nodenames(self):
    """
        Test tokenization of node names.
        """
    self.assertEqual(tgrep.tgrep_tokenize('Robert'), ['Robert'])
    self.assertEqual(tgrep.tgrep_tokenize('/^[Bb]ob/'), ['/^[Bb]ob/'])
    self.assertEqual(tgrep.tgrep_tokenize('*'), ['*'])
    self.assertEqual(tgrep.tgrep_tokenize('__'), ['__'])
    self.assertEqual(tgrep.tgrep_tokenize('N()'), ['N(', ')'])
    self.assertEqual(tgrep.tgrep_tokenize('N(0,)'), ['N(', '0', ',', ')'])
    self.assertEqual(tgrep.tgrep_tokenize('N(0,0)'), ['N(', '0', ',', '0', ')'])
    self.assertEqual(tgrep.tgrep_tokenize('N(0,0,)'), ['N(', '0', ',', '0', ',', ')'])