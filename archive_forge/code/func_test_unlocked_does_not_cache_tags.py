from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_unlocked_does_not_cache_tags(self):
    """Unlocked branches do not cache tags."""
    b1, [rev1, rev2, rev3] = self.make_branch_with_revision_tuple('b', 3)
    b1.tags.set_tag('one', rev1)
    b2 = b1.controldir.open_branch()
    self.assertEqual({'one': rev1}, b1.tags.get_tag_dict())
    b2.tags.set_tag('one', rev2)
    b2.tags.set_tag('two', rev3)
    self.assertEqual({'one': rev2, 'two': rev3}, b1.tags.get_tag_dict())