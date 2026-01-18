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
def test_sprout_controldir_passing_rev_not_source_branch_copies_tags(self):
    builder = self.make_branch_builder('source')
    base_rev = builder.build_commit(message='Base')
    source = builder.get_branch()
    rev_a1 = builder.build_commit(message='Rev A1')
    rev_a2 = builder.build_commit(message='Rev A2')
    rev_a3 = builder.build_commit(message='Rev A3')
    source.set_last_revision_info(1, base_rev)
    rev_b1 = builder.build_commit(message='Rev B1')
    rev_b2 = builder.build_commit(message='Rev B2')
    rev_b3 = builder.build_commit(message='Rev B3')
    source.set_last_revision_info(1, base_rev)
    rev_c1 = builder.build_commit(message='Rev C1')
    rev_c2 = builder.build_commit(message='Rev C2')
    rev_c3 = builder.build_commit(message='Rev C3')
    source.set_last_revision_info(3, rev_a2)
    try:
        source.tags.set_tag('tag-non-ancestry', rev_b2)
    except errors.TagsNotSupported:
        raise TestNotApplicable('Branch format does not support tags ')
    try:
        source.tags.set_tag('tag-absent', b'absent-rev')
    except errors.GhostTagsNotSupported:
        has_ghost_tag = False
    else:
        has_ghost_tag = True
    source.get_config_stack().set('branch.fetch_tags', True)
    dir = source.controldir
    target = dir.sprout(self.get_url('target'), revision_id=rev_c2)
    new_branch = target.open_branch()
    if has_ghost_tag:
        self.assertEqual({'tag-absent': b'absent-rev', 'tag-non-ancestry': rev_b2}, new_branch.tags.get_tag_dict())
    else:
        self.assertEqual({'tag-non-ancestry': rev_b2}, new_branch.tags.get_tag_dict())
    self.assertEqual(sorted([base_rev, rev_b1, rev_b2, rev_c1, rev_c2]), sorted(new_branch.repository.all_revision_ids()))