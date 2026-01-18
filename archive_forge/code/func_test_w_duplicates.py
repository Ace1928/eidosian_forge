import unittest
def test_w_duplicates(self):
    self.assertEqual(self._callFUT([['a'], ['b', 'a']]), ['b', 'a'])