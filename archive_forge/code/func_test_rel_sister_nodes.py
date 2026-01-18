import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_rel_sister_nodes(self):
    """
        Test matching sister nodes in a tree.
        """
    tree = ParentedTree.fromstring('(S (A x) (B x) (C x))')
    self.assertEqual(list(tgrep.tgrep_positions('* $. B', [tree])), [[(0,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* $.. B', [tree])), [[(0,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* $, B', [tree])), [[(2,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* $,, B', [tree])), [[(2,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* $ B', [tree])), [[(0,), (2,)]])