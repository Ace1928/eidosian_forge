import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
class TestHistoryChange(tests.TestCaseWithTransport):

    def setup_a_tree(self):
        tree = self.make_branch_and_tree('tree')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        tree.commit('1a', rev_id=b'1a')
        tree.commit('2a', rev_id=b'2a')
        tree.commit('3a', rev_id=b'3a')
        return tree

    def setup_ab_tree(self):
        tree = self.setup_a_tree()
        tree.set_last_revision(b'1a')
        tree.branch.set_last_revision_info(1, b'1a')
        tree.commit('2b', rev_id=b'2b')
        tree.commit('3b', rev_id=b'3b')
        return tree

    def setup_ac_tree(self):
        tree = self.setup_a_tree()
        tree.set_last_revision(revision.NULL_REVISION)
        tree.branch.set_last_revision_info(0, revision.NULL_REVISION)
        tree.commit('1c', rev_id=b'1c')
        tree.commit('2c', rev_id=b'2c')
        tree.commit('3c', rev_id=b'3c')
        return tree

    def test_all_new(self):
        tree = self.setup_ab_tree()
        old, new = log.get_history_change(b'1a', b'3a', tree.branch.repository)
        self.assertEqual([], old)
        self.assertEqual([b'2a', b'3a'], new)

    def test_all_old(self):
        tree = self.setup_ab_tree()
        old, new = log.get_history_change(b'3a', b'1a', tree.branch.repository)
        self.assertEqual([], new)
        self.assertEqual([b'2a', b'3a'], old)

    def test_null_old(self):
        tree = self.setup_ab_tree()
        old, new = log.get_history_change(revision.NULL_REVISION, b'3a', tree.branch.repository)
        self.assertEqual([], old)
        self.assertEqual([b'1a', b'2a', b'3a'], new)

    def test_null_new(self):
        tree = self.setup_ab_tree()
        old, new = log.get_history_change(b'3a', revision.NULL_REVISION, tree.branch.repository)
        self.assertEqual([], new)
        self.assertEqual([b'1a', b'2a', b'3a'], old)

    def test_diverged(self):
        tree = self.setup_ab_tree()
        old, new = log.get_history_change(b'3a', b'3b', tree.branch.repository)
        self.assertEqual(old, [b'2a', b'3a'])
        self.assertEqual(new, [b'2b', b'3b'])

    def test_unrelated(self):
        tree = self.setup_ac_tree()
        old, new = log.get_history_change(b'3a', b'3c', tree.branch.repository)
        self.assertEqual(old, [b'1a', b'2a', b'3a'])
        self.assertEqual(new, [b'1c', b'2c', b'3c'])

    def test_show_branch_change(self):
        tree = self.setup_ab_tree()
        s = StringIO()
        log.show_branch_change(tree.branch, s, 3, b'3a')
        self.assertContainsRe(s.getvalue(), '[*]{60}\nRemoved Revisions:\n(.|\n)*2a(.|\n)*3a(.|\n)*[*]{60}\n\nAdded Revisions:\n(.|\n)*2b(.|\n)*3b')

    def test_show_branch_change_no_change(self):
        tree = self.setup_ab_tree()
        s = StringIO()
        log.show_branch_change(tree.branch, s, 3, b'3b')
        self.assertEqual(s.getvalue(), 'Nothing seems to have changed\n')

    def test_show_branch_change_no_old(self):
        tree = self.setup_ab_tree()
        s = StringIO()
        log.show_branch_change(tree.branch, s, 2, b'2b')
        self.assertContainsRe(s.getvalue(), 'Added Revisions:')
        self.assertNotContainsRe(s.getvalue(), 'Removed Revisions:')

    def test_show_branch_change_no_new(self):
        tree = self.setup_ab_tree()
        tree.branch.set_last_revision_info(2, b'2b')
        s = StringIO()
        log.show_branch_change(tree.branch, s, 3, b'3b')
        self.assertContainsRe(s.getvalue(), 'Removed Revisions:')
        self.assertNotContainsRe(s.getvalue(), 'Added Revisions:')