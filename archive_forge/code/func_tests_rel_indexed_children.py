import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def tests_rel_indexed_children(self):
    """
        Test matching nodes based on their index in their parent node.
        """
    tree = ParentedTree.fromstring('(S (A x) (B x) (C x))')
    self.assertEqual(list(tgrep.tgrep_positions('* >, S', [tree])), [[(0,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* >1 S', [tree])), [[(0,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* >2 S', [tree])), [[(1,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* >3 S', [tree])), [[(2,)]])
    self.assertEqual(list(tgrep.tgrep_positions("* >' S", [tree])), [[(2,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* >-1 S', [tree])), [[(2,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* >-2 S', [tree])), [[(1,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* >-3 S', [tree])), [[(0,)]])
    tree = ParentedTree.fromstring('(S (D (A x) (B x) (C x)) (E (B x) (C x) (A x)) (F (C x) (A x) (B x)))')
    self.assertEqual(list(tgrep.tgrep_positions('* <, A', [tree])), [[(0,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* <1 A', [tree])), [[(0,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* <2 A', [tree])), [[(2,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* <3 A', [tree])), [[(1,)]])
    self.assertEqual(list(tgrep.tgrep_positions("* <' A", [tree])), [[(1,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* <-1 A', [tree])), [[(1,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* <-2 A', [tree])), [[(2,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* <-3 A', [tree])), [[(0,)]])