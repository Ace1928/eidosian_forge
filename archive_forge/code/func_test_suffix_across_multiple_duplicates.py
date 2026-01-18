import unittest
def test_suffix_across_multiple_duplicates(self):
    O1 = ['x', 'y', 'z']
    O2 = ['q', 'z']
    O3 = [1, 3, 5]
    O4 = ['z']
    self.assertEqual(self._callFUT([O1, O2, O3, O4]), ['x', 'y', 'q', 1, 3, 5, 'z'])