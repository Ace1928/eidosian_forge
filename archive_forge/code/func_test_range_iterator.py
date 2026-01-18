from unittest import TestCase
import patiencediff
from .. import multiparent, tests
def test_range_iterator(self):
    diff = multiparent.MultiParent.from_lines(LINES_1, [LINES_2, LINES_3])
    diff.hunks.append(multiparent.NewText([b'q\n']))
    self.assertEqual([(0, 4, 'parent', (1, 0, 4)), (4, 5, 'parent', (0, 3, 4)), (5, 6, 'new', [b'q\n'])], list(diff.range_iterator()))