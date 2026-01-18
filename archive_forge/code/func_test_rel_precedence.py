import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_rel_precedence(self):
    """
        Test matching nodes based on precedence relations.
        """
    tree = ParentedTree.fromstring('(S (NP (NP (PP x)) (NP (AP x))) (VP (AP (X (PP x)) (Y (AP x)))) (NP (RC (NP (AP x)))))')
    self.assertEqual(list(tgrep.tgrep_positions('* . X', [tree])), [[(0,), (0, 1), (0, 1, 0)]])
    self.assertEqual(list(tgrep.tgrep_positions('* . Y', [tree])), [[(1, 0, 0), (1, 0, 0, 0)]])
    self.assertEqual(list(tgrep.tgrep_positions('* .. X', [tree])), [[(0,), (0, 0), (0, 0, 0), (0, 1), (0, 1, 0)]])
    self.assertEqual(list(tgrep.tgrep_positions('* .. Y', [tree])), [[(0,), (0, 0), (0, 0, 0), (0, 1), (0, 1, 0), (1, 0, 0), (1, 0, 0, 0)]])
    self.assertEqual(list(tgrep.tgrep_positions('* , X', [tree])), [[(1, 0, 1), (1, 0, 1, 0)]])
    self.assertEqual(list(tgrep.tgrep_positions('* , Y', [tree])), [[(2,), (2, 0), (2, 0, 0), (2, 0, 0, 0)]])
    self.assertEqual(list(tgrep.tgrep_positions('* ,, X', [tree])), [[(1, 0, 1), (1, 0, 1, 0), (2,), (2, 0), (2, 0, 0), (2, 0, 0, 0)]])
    self.assertEqual(list(tgrep.tgrep_positions('* ,, Y', [tree])), [[(2,), (2, 0), (2, 0, 0), (2, 0, 0, 0)]])