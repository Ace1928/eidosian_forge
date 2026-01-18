import os
from breezy import branch as _mod_branch
from breezy import errors, osutils
from breezy import revision as _mod_revision
from breezy import tests, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import remote
from breezy.tests import features
from breezy.tests.per_branch import TestCaseWithBranch
def test_sprout_with_unicode_symlink(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    self.requireFeature(features.UnicodeFilenameFeature)
    tree = self.make_branch_and_tree('tree1')
    target = 'Ω'
    link_name = '€link'
    os.symlink(target, 'tree1/' + link_name)
    tree.add([link_name])
    tree.commit('added a link to a Unicode target')
    tree.controldir.sprout('dest')
    self.assertEqual(target, osutils.readlink('dest/' + link_name))
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertEqual(target, tree.get_symlink_target(link_name))
    self.assertEqual(target, tree.basis_tree().get_symlink_target(link_name))