from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_unicode_tag(self):
    tag_name = 'ば'
    b1, [revid] = self.make_branch_with_revision_tuple('b', 1)
    b1.tags.set_tag(tag_name, revid)
    self.assertEqual(b1.tags.lookup_tag(tag_name), revid)