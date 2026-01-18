import breezy.branch
from breezy import branch as _mod_branch
from breezy import check, controldir, errors, gpg, osutils
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import transport, ui, urlutils, workingtree
from breezy.bzr import bzrdir as _mod_bzrdir
from breezy.bzr.remote import (RemoteBzrDir, RemoteBzrDirFormat,
from breezy.tests import (ChrootedTestCase, TestNotApplicable, TestSkipped,
from breezy.tests.per_controldir import TestCaseWithControlDir
from breezy.transport.local import LocalTransport
from breezy.ui import CannedInputUIFactory
def test_sprout_controldir_passing_source_branch_with_absent_tag(self):
    builder = self.make_branch_builder('source')
    builder.build_commit(message='Rev 1')
    source = builder.get_branch()
    try:
        source.tags.set_tag('tag-a', b'missing-rev')
    except (errors.TagsNotSupported, errors.GhostTagsNotSupported):
        raise TestNotApplicable('Branch format does not support tags or tags referencing missing revisions.')
    dir = source.controldir
    target = dir.sprout(self.get_url('target'), source_branch=source)
    new_branch = target.open_branch()
    self.assertEqual(b'missing-rev', new_branch.tags.lookup_tag('tag-a'))