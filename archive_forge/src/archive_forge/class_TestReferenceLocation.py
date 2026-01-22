import errno
import os
from io import StringIO
from ... import branch as _mod_branch
from ... import config, controldir, errors, merge, osutils
from ... import revision as _mod_revision
from ... import tests, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import bzrdir
from ...bzr.conflicts import ConflictList, ContentsConflict, TextConflict
from ...bzr.inventory import Inventory
from ...bzr.workingtree import InventoryWorkingTree
from ...errors import PathsNotVersionedError, UnsupportedOperation
from ...mutabletree import MutableTree
from ...osutils import getcwd, pathjoin, supports_symlinks
from ...tree import TreeDirectory, TreeFile, TreeLink
from ...workingtree import SettingFileIdUnsupported, WorkingTree
from .. import TestNotApplicable, TestSkipped, features
from . import TestCaseWithWorkingTree
class TestReferenceLocation(TestCaseWithWorkingTree):

    def test_reference_parent(self):
        tree = self.make_branch_and_tree('tree')
        subtree = self.make_branch_and_tree('tree/subtree')
        subtree.commit('a change')
        try:
            tree.add_reference(subtree)
        except errors.UnsupportedOperation:
            raise tests.TestNotApplicable('Tree cannot hold references.')
        if not getattr(tree.branch._format, 'supports_reference_locations', False):
            raise tests.TestNotApplicable('Branch cannot hold reference locations.')
        tree.commit('Add reference.')
        reference_parent = tree.reference_parent(urlutils.relative_url(urlutils.strip_segment_parameters(tree.branch.user_url), urlutils.strip_segment_parameters(subtree.branch.user_url)))
        self.assertEqual(subtree.branch.user_url, reference_parent.user_url)

    def test_reference_parent_accepts_possible_transports(self):
        tree = self.make_branch_and_tree('tree')
        subtree = self.make_branch_and_tree('tree/subtree')
        subtree.commit('a change')
        try:
            tree.add_reference(subtree)
        except errors.UnsupportedOperation:
            raise tests.TestNotApplicable('Tree cannot hold references.')
        if not getattr(tree.branch._format, 'supports_reference_locations', False):
            raise tests.TestNotApplicable('Branch cannot hold reference locations.')
        tree.commit('Add reference')
        reference_parent = tree.reference_parent(urlutils.relative_url(urlutils.strip_segment_parameters(tree.branch.user_url), urlutils.strip_segment_parameters(subtree.branch.user_url)), possible_transports=[subtree.controldir.root_transport])

    def test_get_reference_info(self):
        tree = self.make_branch_and_tree('branch')
        try:
            loc = tree.get_reference_info('file')
        except errors.UnsupportedOperation:
            raise tests.TestNotApplicable('Branch cannot hold references.')
        self.assertIs(None, loc)

    def test_set_reference_info(self):
        self.make_tree_with_reference('branch', 'path/to/location')

    def test_set_get_reference_info(self):
        tree = self.make_tree_with_reference('branch', 'path/to/location')
        tree = WorkingTree.open('branch')
        branch_location = tree.get_reference_info('path/to/file')
        self.assertEqual(urlutils.join(urlutils.strip_segment_parameters(tree.branch.user_url), 'path/to/location'), urlutils.join(urlutils.strip_segment_parameters(tree.branch.user_url), branch_location))

    def test_set_null_reference_info(self):
        tree = self.make_branch_and_tree('branch')
        self.build_tree(['branch/file'])
        tree.add(['file'])
        try:
            tree.set_reference_info('file', 'path/to/location')
        except errors.UnsupportedOperation:
            raise tests.TestNotApplicable('Branch cannot hold references.')
        tree.set_reference_info('file', None)
        branch_location = tree.get_reference_info('file')
        self.assertIs(None, branch_location)

    def test_set_null_reference_info_when_null(self):
        tree = self.make_branch_and_tree('branch')
        try:
            branch_location = tree.get_reference_info('file')
        except errors.UnsupportedOperation:
            raise tests.TestNotApplicable('Branch cannot hold references.')
        self.assertIs(None, branch_location)
        self.build_tree(['branch/file'])
        tree.add(['file'])
        try:
            tree.set_reference_info('file', None)
        except errors.UnsupportedOperation:
            raise tests.TestNotApplicable('Branch cannot hold references.')

    def make_tree_with_reference(self, location, reference_location):
        tree = self.make_branch_and_tree(location)
        self.build_tree([os.path.join(location, name) for name in ['path/', 'path/to/', 'path/to/file']])
        tree.add(['path', 'path/to', 'path/to/file'])
        try:
            tree.set_reference_info('path/to/file', reference_location)
        except errors.UnsupportedOperation:
            raise tests.TestNotApplicable('Branch cannot hold references.')
        tree.commit('commit reference')
        return tree

    def test_reference_parent_from_reference_info_(self):
        referenced_branch = self.make_branch('reference_branch')
        tree = self.make_tree_with_reference('branch', referenced_branch.base)
        parent = tree.reference_parent('path/to/file')
        self.assertEqual(parent.base, referenced_branch.base)

    def test_branch_relative_reference_location(self):
        tree = self.make_tree_with_reference('branch', '../reference_branch')
        referenced_branch = self.make_branch('reference_branch')
        parent = tree.reference_parent('path/to/file')
        self.assertEqual(parent.base, referenced_branch.base)

    def test_sprout_copies_reference_location(self):
        tree = self.make_tree_with_reference('branch', '../reference')
        new_tree = tree.branch.controldir.sprout('new-branch').open_workingtree()
        self.assertEqual(urlutils.join(urlutils.strip_segment_parameters(tree.branch.user_url), '../reference'), urlutils.join(urlutils.strip_segment_parameters(new_tree.branch.user_url), new_tree.get_reference_info('path/to/file')))

    def test_clone_copies_reference_location(self):
        tree = self.make_tree_with_reference('branch', '../reference')
        new_tree = tree.controldir.clone('new-branch').open_workingtree()
        self.assertEqual(urlutils.join(urlutils.strip_segment_parameters(tree.branch.user_url), '../reference'), urlutils.join(urlutils.strip_segment_parameters(new_tree.branch.user_url), new_tree.get_reference_info('path/to/file')))

    def test_copied_locations_are_rebased(self):
        tree = self.make_tree_with_reference('branch', 'reference')
        new_tree = tree.controldir.sprout('branch/new-branch').open_workingtree()
        self.assertEqual(urlutils.join(urlutils.strip_segment_parameters(tree.branch.user_url), 'reference'), urlutils.join(urlutils.strip_segment_parameters(new_tree.branch.user_url), new_tree.get_reference_info('path/to/file')))

    def test_update_references_retains_old_references(self):
        tree = self.make_tree_with_reference('branch', 'reference')
        new_tree = self.make_tree_with_reference('new_branch', 'reference2')
        new_tree.branch.update_references(tree.branch)
        self.assertEqual(urlutils.join(urlutils.strip_segment_parameters(tree.branch.user_url), 'reference'), urlutils.join(urlutils.strip_segment_parameters(tree.branch.user_url), tree.get_reference_info('path/to/file')))

    def test_update_references_retains_known_references(self):
        tree = self.make_tree_with_reference('branch', 'reference')
        new_tree = self.make_tree_with_reference('new_branch', 'reference2')
        new_tree.branch.update_references(tree.branch)
        self.assertEqual(urlutils.join(urlutils.strip_segment_parameters(tree.branch.user_url), 'reference'), urlutils.join(urlutils.strip_segment_parameters(tree.branch.user_url), tree.get_reference_info('path/to/file')))

    def test_update_references_skips_known_references(self):
        tree = self.make_tree_with_reference('branch', 'reference')
        new_tree = tree.controldir.sprout('branch/new-branch').open_workingtree()
        self.build_tree(['branch/new-branch/foo'])
        new_tree.add('foo')
        new_tree.set_reference_info('foo', '../foo')
        new_tree.branch.update_references(tree.branch)
        self.assertEqual(urlutils.join(urlutils.strip_segment_parameters(tree.branch.user_url), 'reference'), urlutils.join(urlutils.strip_segment_parameters(tree.branch.user_url), tree.get_reference_info('path/to/file')))

    def test_pull_updates_references(self):
        tree = self.make_tree_with_reference('branch', 'reference')
        new_tree = tree.controldir.sprout('branch/new-branch').open_workingtree()
        self.build_tree(['branch/new-branch/foo'])
        new_tree.add('foo')
        new_tree.set_reference_info('foo', '../foo')
        new_tree.commit('set reference')
        tree.pull(new_tree.branch)
        self.assertEqual(urlutils.join(urlutils.strip_segment_parameters(new_tree.branch.user_url), '../foo'), urlutils.join(tree.branch.user_url, tree.get_reference_info('foo')))

    def test_push_updates_references(self):
        tree = self.make_tree_with_reference('branch', 'reference')
        new_tree = tree.controldir.sprout('branch/new-branch').open_workingtree()
        self.build_tree(['branch/new-branch/foo'])
        new_tree.add(['foo'])
        new_tree.set_reference_info('foo', '../foo')
        new_tree.commit('add reference')
        tree.pull(new_tree.branch)
        tree.update()
        self.assertEqual(urlutils.join(urlutils.strip_segment_parameters(new_tree.branch.user_url), '../foo'), urlutils.join(urlutils.strip_segment_parameters(tree.branch.user_url), tree.get_reference_info('foo')))

    def test_merge_updates_references(self):
        orig_tree = self.make_tree_with_reference('branch', 'reference')
        tree = orig_tree.controldir.sprout('tree').open_workingtree()
        tree.commit('foo')
        orig_tree.pull(tree.branch)
        checkout = orig_tree.branch.create_checkout('checkout', lightweight=True)
        checkout.commit('bar')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        merger = merge.Merger.from_revision_ids(tree, orig_tree.branch.last_revision(), other_branch=orig_tree.branch)
        merger.merge_type = merge.Merge3Merger
        merger.do_merge()
        self.assertEqual(urlutils.join(urlutils.strip_segment_parameters(orig_tree.branch.user_url), 'reference'), urlutils.join(urlutils.strip_segment_parameters(tree.branch.user_url), tree.get_reference_info('path/to/file')))