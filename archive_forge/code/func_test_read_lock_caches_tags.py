from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_read_lock_caches_tags(self):
    """Tags are read from a branch only once during a read-lock."""
    b1, [rev1, rev2, rev3] = self.make_branch_with_revision_tuple('b', 3)
    b1.tags.set_tag('one', rev1)
    b2 = controldir.ControlDir.open('b').open_branch()
    b1.lock_read()
    self.assertEqual({'one': rev1}, b1.tags.get_tag_dict())
    b2.tags.set_tag('one', rev2)
    b2.tags.set_tag('two', rev3)
    self.assertEqual({'one': rev1}, b1.tags.get_tag_dict())
    b1.unlock()
    self.assertEqual({'one': rev2, 'two': rev3}, b1.tags.get_tag_dict())