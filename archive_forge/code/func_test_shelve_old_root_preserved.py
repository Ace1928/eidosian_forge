import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def test_shelve_old_root_preserved(self):
    tree1 = self.make_branch_and_tree('tree1')
    tree1.commit('add root')
    tree1_root_id = tree1.path2id('')
    tree2 = self.make_branch_and_tree('tree2')
    rev2 = tree2.commit('add root')
    self.assertNotEqual(tree1_root_id, tree2.path2id(''))
    tree1.merge_from_branch(tree2.branch, from_revision=revision.NULL_REVISION)
    tree1.commit('merging in tree2')
    self.assertEqual(tree1_root_id, tree1.path2id(''))
    e = self.assertRaises(AssertionError, self.assertRaises, errors.InconsistentDelta, self.shelve_all, tree1, rev2)
    self.assertContainsRe('InconsistentDelta not raised', str(e))