import unittest
from collections import Counter
from low_index import *
def test_o9_15405_9(self):
    reps = permutation_reps(2, [], ['aaaaabbbaabbbaaaaabbbaabbbaaaaaBBBBBBBB'], 9)
    degrees = Counter([len(rep[0]) for rep in reps])
    self.assertEqual(degrees[1], 1)
    self.assertEqual(degrees[2], 1)
    self.assertEqual(degrees[3], 1)
    self.assertEqual(degrees[4], 1)
    self.assertEqual(degrees[5], 3)
    self.assertEqual(degrees[6], 3)
    self.assertEqual(degrees[7], 9)
    self.assertEqual(degrees[8], 5)
    self.assertEqual(degrees[9], 14)
    self.assertIn([[0], [0]], reps)
    self.assertIn([[0, 2, 4, 1, 5, 3], [1, 0, 5, 4, 2, 3]], reps)