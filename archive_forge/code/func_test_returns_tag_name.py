from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_returns_tag_name(self):

    def get_tag_name(br, revid):
        return 'foo'
    branch.Branch.hooks.install_named_hook('automatic_tag_name', get_tag_name, 'get tag name foo')
    self.assertEqual('foo', self.branch.automatic_tag_name(self.branch.last_revision()))