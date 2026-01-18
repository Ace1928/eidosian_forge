import os
import sys
from unicodedata import normalize
from .. import osutils
from ..osutils import pathjoin
from . import TestCase, TestCaseWithTransport, TestSkipped
def test_platform(self):
    files = [a_circle_c + '.1', a_dots_c + '.2', z_umlat_c + '.3']
    try:
        self.build_tree(files)
    except UnicodeError:
        raise TestSkipped('filesystem cannot create unicode files')
    if sys.platform == 'darwin':
        expected = sorted([a_circle_d + '.1', a_dots_d + '.2', z_umlat_d + '.3'])
    else:
        expected = sorted(files)
    present = sorted(os.listdir('.'))
    self.assertEqual(expected, present)