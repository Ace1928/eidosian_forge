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
def test_sprout_controldir_branch_with_tags(self):
    builder = self.make_branch_builder('source')
    source, rev1, rev2 = fixtures.build_branch_with_non_ancestral_rev(builder)
    try:
        source.tags.set_tag('tag-a', rev2)
    except errors.TagsNotSupported:
        raise TestNotApplicable('Branch format does not support tags.')
    source.get_config_stack().set('branch.fetch_tags', True)
    dir = source.controldir
    target = dir.sprout(self.get_url('target'))
    new_branch = target.open_branch()
    self.assertEqual(rev2, new_branch.tags.lookup_tag('tag-a'))
    new_branch.repository.get_revision(rev2)