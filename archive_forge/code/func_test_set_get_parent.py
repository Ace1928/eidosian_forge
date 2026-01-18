import sys
import breezy.errors
from breezy import urlutils
from breezy.osutils import getcwd
from breezy.tests import TestCaseWithTransport, TestNotApplicable, TestSkipped
def test_set_get_parent(self):
    """Set, re-get and reset the parent"""
    b = self.make_branch('subdir')
    url = 'http://example.com/bzr/bzr.dev'
    b.set_parent(url)
    self.assertEqual(url, b.get_parent())
    self.assertEqual(url, b._get_parent_location())
    b.set_parent(None)
    self.assertEqual(None, b.get_parent())
    b.set_parent('../other_branch')
    expected_parent = urlutils.join(self.get_url('subdir'), '../other_branch')
    self.assertEqual(expected_parent, b.get_parent())
    path = urlutils.join(self.get_url('subdir'), '../yanb')
    b.set_parent(path)
    self.assertEqual('../yanb', b._get_parent_location())
    self.assertEqual(path, b.get_parent())
    self.assertRaises(urlutils.InvalidURL, b.set_parent, 'µ')
    b.set_parent(urlutils.escape('µ'))
    self.assertEqual('%C2%B5', b._get_parent_location())
    self.assertEqual(b.base + '%C2%B5', b.get_parent())
    if sys.platform == 'win32':
        pass
    else:
        b.lock_write()
        b._set_parent_location('/local/abs/path')
        b.unlock()
        self.assertEqual('file:///local/abs/path', b.get_parent())