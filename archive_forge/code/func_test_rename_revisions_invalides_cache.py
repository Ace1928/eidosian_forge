from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_rename_revisions_invalides_cache(self):
    b, revids = self.make_write_locked_branch_with_one_tag()
    b.tags.rename_revisions({revids[0]: revids[1]})
    self.assertEqual({'one': revids[1]}, b.tags.get_tag_dict())