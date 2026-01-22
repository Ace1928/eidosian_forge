import os
from breezy import branch, conflicts, controldir, errors, mutabletree, osutils
from breezy import revision as _mod_revision
from breezy import tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.commit import CannotCommitSelectedFileMerge, PointlessCommit
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.tests.testui import ProgressRecordingUIFactory
class TestCommitProgress(TestCaseWithWorkingTree):

    def setUp(self):
        super().setUp()
        ui.ui_factory = ProgressRecordingUIFactory()

    def test_commit_progress_steps(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a', 'b', 'c'])
        tree.add(['a', 'b', 'c'])
        tree.commit('first post')
        with open('b', 'w') as f:
            f.write('new content')
        factory = ProgressRecordingUIFactory()
        ui.ui_factory = factory
        tree.commit('second post', specific_files=['b'])
        self.assertEqual([('update', 1, 5, 'Collecting changes [0] - Stage'), ('update', 1, 5, 'Collecting changes [1] - Stage'), ('update', 2, 5, 'Saving data locally - Stage'), ('update', 3, 5, 'Running pre_commit hooks - Stage'), ('update', 4, 5, 'Updating the working tree - Stage'), ('update', 5, 5, 'Running post_commit hooks - Stage')], factory._calls)

    def test_commit_progress_shows_post_hook_names(self):
        tree = self.make_branch_and_tree('.')
        factory = ProgressRecordingUIFactory()
        ui.ui_factory = factory

        def a_hook(_, _2, _3, _4, _5, _6):
            pass
        branch.Branch.hooks.install_named_hook('post_commit', a_hook, 'hook name')
        tree.commit('first post')
        self.assertEqual([('update', 1, 5, 'Collecting changes [0] - Stage'), ('update', 1, 5, 'Collecting changes [1] - Stage'), ('update', 2, 5, 'Saving data locally - Stage'), ('update', 3, 5, 'Running pre_commit hooks - Stage'), ('update', 4, 5, 'Updating the working tree - Stage'), ('update', 5, 5, 'Running post_commit hooks - Stage'), ('update', 5, 5, 'Running post_commit hooks [hook name] - Stage')], factory._calls)

    def test_commit_progress_shows_pre_hook_names(self):
        tree = self.make_branch_and_tree('.')
        factory = ProgressRecordingUIFactory()
        ui.ui_factory = factory

        def a_hook(_, _2, _3, _4, _5, _6, _7, _8):
            pass
        branch.Branch.hooks.install_named_hook('pre_commit', a_hook, 'hook name')
        tree.commit('first post')
        self.assertEqual([('update', 1, 5, 'Collecting changes [0] - Stage'), ('update', 1, 5, 'Collecting changes [1] - Stage'), ('update', 2, 5, 'Saving data locally - Stage'), ('update', 3, 5, 'Running pre_commit hooks - Stage'), ('update', 3, 5, 'Running pre_commit hooks [hook name] - Stage'), ('update', 4, 5, 'Updating the working tree - Stage'), ('update', 5, 5, 'Running post_commit hooks - Stage')], factory._calls)

    def test_start_commit_hook(self):
        """Make sure a start commit hook can modify the tree that is
        committed."""

        def start_commit_hook_adds_file(tree):
            with open(tree.abspath('newfile'), 'w') as f:
                f.write('data')
            tree.add(['newfile'])

        def restoreDefaults():
            mutabletree.MutableTree.hooks['start_commit'] = []
        self.addCleanup(restoreDefaults)
        tree = self.make_branch_and_tree('.')
        mutabletree.MutableTree.hooks.install_named_hook('start_commit', start_commit_hook_adds_file, None)
        revid = tree.commit('first post')
        committed_tree = tree.basis_tree()
        self.assertTrue(committed_tree.has_filename('newfile'))

    def test_post_commit_hook(self):
        """Make sure a post_commit hook is called after a commit."""

        def post_commit_hook_test_params(params):
            self.assertTrue(isinstance(params, mutabletree.PostCommitHookParams))
            self.assertTrue(isinstance(params.mutable_tree, mutabletree.MutableTree))
            with open(tree.abspath('newfile'), 'w') as f:
                f.write('data')
            params.mutable_tree.add(['newfile'])
        tree = self.make_branch_and_tree('.')
        mutabletree.MutableTree.hooks.install_named_hook('post_commit', post_commit_hook_test_params, None)
        self.assertFalse(tree.has_filename('newfile'))
        revid = tree.commit('first post')
        self.assertTrue(tree.has_filename('newfile'))
        committed_tree = tree.basis_tree()
        self.assertFalse(committed_tree.has_filename('newfile'))