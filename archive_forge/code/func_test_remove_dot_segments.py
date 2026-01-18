from collections import defaultdict
import unittest
from lazr.uri import (
def test_remove_dot_segments(self):
    self.assertEqual(remove_dot_segments('/a/b/c/./../../g'), '/a/g')
    self.assertEqual(remove_dot_segments('mid/content=5/../6'), 'mid/6')