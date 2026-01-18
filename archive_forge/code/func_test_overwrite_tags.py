import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_overwrite_tags(self):
    """--overwrite-tags only overwrites tags, not revisions."""
    from_tree = self.make_branch_and_tree('from')
    from_tree.branch.tags.set_tag('mytag', b'somerevid')
    to_tree = self.make_branch_and_tree('to')
    to_tree.branch.tags.set_tag('mytag', b'anotherrevid')
    revid1 = to_tree.commit('my commit')
    out = self.run_bzr(['push', '-d', 'from', 'to'])
    self.assertEqual(out, ('Conflicting tags:\n    mytag\n', 'No new revisions to push.\n'))
    out = self.run_bzr(['push', '-d', 'from', '--overwrite-tags', 'to'])
    self.assertEqual(out, ('', '1 tag updated.\n'))
    self.assertEqual(to_tree.branch.tags.lookup_tag('mytag'), b'somerevid')
    self.assertEqual(to_tree.branch.last_revision(), revid1)