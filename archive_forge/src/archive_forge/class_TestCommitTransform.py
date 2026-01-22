import codecs
import errno
import os
import sys
import time
from io import BytesIO, StringIO
import fastbencode as bencode
from .. import filters, osutils
from .. import revision as _mod_revision
from .. import rules, tests, trace, transform, urlutils
from ..bzr import generate_ids
from ..bzr.conflicts import (DeletingParent, DuplicateEntry, DuplicateID,
from ..controldir import ControlDir
from ..diff import show_diff_trees
from ..errors import (DuplicateKey, ExistingLimbo, ExistingPendingDeletion,
from ..merge import Merge3Merger, Merger
from ..mutabletree import MutableTree
from ..osutils import file_kind, pathjoin
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..transport import FileExists
from . import TestCaseInTempDir, TestSkipped, features
from .features import HardlinkFeature, SymlinkFeature
class TestCommitTransform(tests.TestCaseWithTransport):

    def get_branch(self):
        tree = self.make_branch_and_tree('tree')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        tree.commit('empty commit')
        return tree.branch

    def get_branch_and_transform(self):
        branch = self.get_branch()
        tt = branch.basis_tree().preview_transform()
        self.addCleanup(tt.finalize)
        return (branch, tt)

    def test_commit_wrong_basis(self):
        branch = self.get_branch()
        basis = branch.repository.revision_tree(_mod_revision.NULL_REVISION)
        tt = basis.preview_transform()
        self.addCleanup(tt.finalize)
        e = self.assertRaises(ValueError, tt.commit, branch, '')
        self.assertEqual('TreeTransform not based on branch basis: null:', str(e))

    def test_empy_commit(self):
        branch, tt = self.get_branch_and_transform()
        rev = tt.commit(branch, 'my message')
        self.assertEqual(2, branch.revno())
        repo = branch.repository
        self.assertEqual('my message', repo.get_revision(rev).message)

    def test_merge_parents(self):
        branch, tt = self.get_branch_and_transform()
        tt.commit(branch, 'my message', [b'rev1b', b'rev1c'])
        self.assertEqual([b'rev1b', b'rev1c'], branch.basis_tree().get_parent_ids()[1:])

    def test_first_commit(self):
        branch = self.make_branch('branch')
        branch.lock_write()
        self.addCleanup(branch.unlock)
        tt = branch.basis_tree().preview_transform()
        self.addCleanup(tt.finalize)
        tt.new_directory('', ROOT_PARENT, b'TREE_ROOT')
        tt.commit(branch, 'my message')
        self.assertEqual([], branch.basis_tree().get_parent_ids())
        self.assertNotEqual(_mod_revision.NULL_REVISION, branch.last_revision())

    def test_first_commit_with_merge_parents(self):
        branch = self.make_branch('branch')
        branch.lock_write()
        self.addCleanup(branch.unlock)
        tt = branch.basis_tree().preview_transform()
        self.addCleanup(tt.finalize)
        e = self.assertRaises(ValueError, tt.commit, branch, 'my message', [b'rev1b-id'])
        self.assertEqual('Cannot supply merge parents for first commit.', str(e))
        self.assertEqual(_mod_revision.NULL_REVISION, branch.last_revision())

    def test_add_files(self):
        branch, tt = self.get_branch_and_transform()
        tt.new_file('file', tt.root, [b'contents'], b'file-id')
        trans_id = tt.new_directory('dir', tt.root, b'dir-id')
        if SymlinkFeature(self.test_dir).available():
            tt.new_symlink('symlink', trans_id, 'target', b'symlink-id')
        tt.commit(branch, 'message')
        tree = branch.basis_tree()
        self.assertEqual('file', tree.id2path(b'file-id'))
        self.assertEqual(b'contents', tree.get_file_text('file'))
        self.assertEqual('dir', tree.id2path(b'dir-id'))
        if SymlinkFeature(self.test_dir).available():
            self.assertEqual('dir/symlink', tree.id2path(b'symlink-id'))
            self.assertEqual('target', tree.get_symlink_target('dir/symlink'))

    def test_add_unversioned(self):
        branch, tt = self.get_branch_and_transform()
        tt.new_file('file', tt.root, [b'contents'])
        self.assertRaises(StrictCommitFailed, tt.commit, branch, 'message', strict=True)

    def test_modify_strict(self):
        branch, tt = self.get_branch_and_transform()
        tt.new_file('file', tt.root, [b'contents'], b'file-id')
        tt.commit(branch, 'message', strict=True)
        tt = branch.basis_tree().preview_transform()
        self.addCleanup(tt.finalize)
        trans_id = tt.trans_id_file_id(b'file-id')
        tt.delete_contents(trans_id)
        tt.create_file([b'contents'], trans_id)
        tt.commit(branch, 'message', strict=True)

    def test_commit_malformed(self):
        """Committing a malformed transform should raise an exception.

        In this case, we are adding a file without adding its parent.
        """
        branch, tt = self.get_branch_and_transform()
        parent_id = tt.trans_id_file_id(b'parent-id')
        tt.new_file('file', parent_id, [b'contents'], b'file-id')
        self.assertRaises(MalformedTransform, tt.commit, branch, 'message')

    def test_commit_rich_revision_data(self):
        branch, tt = self.get_branch_and_transform()
        rev_id = tt.commit(branch, 'message', timestamp=1, timezone=43201, committer='me <me@example.com>', revprops={'foo': 'bar'}, revision_id=b'revid-1', authors=['Author1 <author1@example.com>', 'Author2 <author2@example.com>'])
        self.assertEqual(b'revid-1', rev_id)
        revision = branch.repository.get_revision(rev_id)
        self.assertEqual(1, revision.timestamp)
        self.assertEqual(43201, revision.timezone)
        self.assertEqual('me <me@example.com>', revision.committer)
        self.assertEqual(['Author1 <author1@example.com>', 'Author2 <author2@example.com>'], revision.get_apparent_authors())
        del revision.properties['authors']
        self.assertEqual({'foo': 'bar', 'branch-nick': 'tree'}, revision.properties)

    def test_no_explicit_revprops(self):
        branch, tt = self.get_branch_and_transform()
        rev_id = tt.commit(branch, 'message', authors=['Author1 <author1@example.com>', 'Author2 <author2@example.com>'])
        revision = branch.repository.get_revision(rev_id)
        self.assertEqual(['Author1 <author1@example.com>', 'Author2 <author2@example.com>'], revision.get_apparent_authors())
        self.assertEqual('tree', revision.properties['branch-nick'])