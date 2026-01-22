from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
class DummyForeignVcsTests(tests.TestCaseWithTransport):
    """Very basic test for DummyForeignVcs."""

    def setUp(self):
        super().setUp()
        register_dummy_foreign_for_test(self)

    def test_create(self):
        """Test we can create dummies."""
        self.make_branch_and_tree('d', format=DummyForeignVcsDirFormat())
        dir = controldir.ControlDir.open('d')
        self.assertEqual(b'A Dummy VCS Dir', dir._format.get_format_string())
        dir.open_repository()
        dir.open_branch()
        dir.open_workingtree()

    def test_sprout(self):
        """Test we can clone dummies and that the format is not preserved."""
        self.make_branch_and_tree('d', format=DummyForeignVcsDirFormat())
        dir = controldir.ControlDir.open('d')
        newdir = dir.sprout('e')
        self.assertNotEqual(b'A Dummy VCS Dir', newdir._format.get_format_string())

    def test_push_not_supported(self):
        source_tree = self.make_branch_and_tree('source')
        target_tree = self.make_branch_and_tree('target', format=DummyForeignVcsDirFormat())
        self.assertRaises(errors.NoRoundtrippingSupport, source_tree.branch.push, target_tree.branch)

    def test_lossy_push_empty(self):
        source_tree = self.make_branch_and_tree('source')
        target_tree = self.make_branch_and_tree('target', format=DummyForeignVcsDirFormat())
        pushresult = source_tree.branch.push(target_tree.branch, lossy=True)
        self.assertEqual(revision.NULL_REVISION, pushresult.old_revid)
        self.assertEqual(revision.NULL_REVISION, pushresult.new_revid)
        self.assertEqual({}, pushresult.revidmap)

    def test_lossy_push_simple(self):
        source_tree = self.make_branch_and_tree('source')
        self.build_tree(['source/a', 'source/b'])
        source_tree.add(['a', 'b'])
        revid1 = source_tree.commit('msg')
        target_tree = self.make_branch_and_tree('target', format=DummyForeignVcsDirFormat())
        target_tree.branch.lock_write()
        try:
            pushresult = source_tree.branch.push(target_tree.branch, lossy=True)
        finally:
            target_tree.branch.unlock()
        self.assertEqual(revision.NULL_REVISION, pushresult.old_revid)
        self.assertEqual({revid1: target_tree.branch.last_revision()}, pushresult.revidmap)
        self.assertEqual(pushresult.revidmap[revid1], pushresult.new_revid)