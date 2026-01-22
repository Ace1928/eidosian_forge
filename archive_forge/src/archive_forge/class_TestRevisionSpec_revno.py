import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
class TestRevisionSpec_revno(TestRevisionSpec):

    def test_positive_int(self):
        self.assertInHistoryIs(0, b'null:', '0')
        self.assertInHistoryIs(1, b'r1', '1')
        self.assertInHistoryIs(2, b'r2', '2')
        self.assertInvalid('3')

    def test_dotted_decimal(self):
        self.assertInHistoryIs(None, b'alt_r2', '1.1.1')
        self.assertInvalid('1.1.123')

    def test_negative_int(self):
        self.assertInHistoryIs(2, b'r2', '-1')
        self.assertInHistoryIs(1, b'r1', '-2')
        self.assertInHistoryIs(1, b'r1', '-3')
        self.assertInHistoryIs(1, b'r1', '-4')
        self.assertInHistoryIs(1, b'r1', '-100')

    def test_positive(self):
        self.assertInHistoryIs(0, b'null:', 'revno:0')
        self.assertInHistoryIs(1, b'r1', 'revno:1')
        self.assertInHistoryIs(2, b'r2', 'revno:2')
        self.assertInvalid('revno:3')

    def test_negative(self):
        self.assertInHistoryIs(2, b'r2', 'revno:-1')
        self.assertInHistoryIs(1, b'r1', 'revno:-2')
        self.assertInHistoryIs(1, b'r1', 'revno:-3')
        self.assertInHistoryIs(1, b'r1', 'revno:-4')

    def test_invalid_number(self):
        try:
            int('X')
        except ValueError as e:
            self.assertInvalid('revno:X', extra='\n' + str(e))
        else:
            self.fail()

    def test_missing_number_and_branch(self):
        self.assertInvalid('revno::', extra='\ncannot have an empty revno and no branch')

    def test_invalid_number_with_branch(self):
        try:
            int('X')
        except ValueError as e:
            self.assertInvalid('revno:X:tree2', extra='\n' + str(e))
        else:
            self.fail()

    def test_non_exact_branch(self):
        spec = RevisionSpec.from_string('revno:2:tree2/a')
        self.assertRaises(errors.NotBranchError, spec.in_history, self.tree.branch)

    def test_with_branch(self):
        revinfo = self.get_in_history('revno:2:tree2')
        self.assertNotEqual(self.tree.branch.base, revinfo.branch.base)
        self.assertEqual(self.tree2.branch.base, revinfo.branch.base)
        self.assertEqual(2, revinfo.revno)
        self.assertEqual(b'alt_r2', revinfo.rev_id)

    def test_int_with_branch(self):
        revinfo = self.get_in_history('2:tree2')
        self.assertNotEqual(self.tree.branch.base, revinfo.branch.base)
        self.assertEqual(self.tree2.branch.base, revinfo.branch.base)
        self.assertEqual(2, revinfo.revno)
        self.assertEqual(b'alt_r2', revinfo.rev_id)

    def test_with_url(self):
        url = self.get_url() + '/tree2'
        revinfo = self.get_in_history('revno:2:{}'.format(url))
        self.assertNotEqual(self.tree.branch.base, revinfo.branch.base)
        self.assertEqual(self.tree2.branch.base, revinfo.branch.base)
        self.assertEqual(2, revinfo.revno)
        self.assertEqual(b'alt_r2', revinfo.rev_id)

    def test_negative_with_url(self):
        url = self.get_url() + '/tree2'
        revinfo = self.get_in_history('revno:-1:{}'.format(url))
        self.assertNotEqual(self.tree.branch.base, revinfo.branch.base)
        self.assertEqual(self.tree2.branch.base, revinfo.branch.base)
        self.assertEqual(2, revinfo.revno)
        self.assertEqual(b'alt_r2', revinfo.rev_id)

    def test_different_history_lengths(self):
        self.tree2.commit('three', rev_id=b'r3')
        self.assertInHistoryIs(3, b'r3', 'revno:3:tree2')
        self.assertInHistoryIs(3, b'r3', 'revno:-1:tree2')

    def test_invalid_branch(self):
        self.assertRaises(errors.NotBranchError, self.get_in_history, 'revno:-1:tree3')

    def test_invalid_revno_in_branch(self):
        self.tree.commit('three', rev_id=b'r3')
        self.assertInvalid('revno:3:tree2')

    def test_revno_n_path(self):
        """Old revno:N:path tests"""
        wta = self.make_branch_and_tree('a')
        ba = wta.branch
        wta.commit('Commit one', rev_id=b'a@r-0-1')
        wta.commit('Commit two', rev_id=b'a@r-0-2')
        wta.commit('Commit three', rev_id=b'a@r-0-3')
        wtb = self.make_branch_and_tree('b')
        bb = wtb.branch
        wtb.commit('Commit one', rev_id=b'b@r-0-1')
        wtb.commit('Commit two', rev_id=b'b@r-0-2')
        wtb.commit('Commit three', rev_id=b'b@r-0-3')
        self.assertEqual((1, b'a@r-0-1'), spec_in_history('revno:1:a/', ba))
        self.assertEqual((1, b'a@r-0-1'), spec_in_history('revno:1:a/', None))
        self.assertEqual((1, b'a@r-0-1'), spec_in_history('revno:1:a/', bb))
        self.assertEqual((2, b'b@r-0-2'), spec_in_history('revno:2:b/', None))

    def test_as_revision_id(self):
        self.assertAsRevisionId(b'null:', '0')
        self.assertAsRevisionId(b'r1', '1')
        self.assertAsRevisionId(b'r2', '2')
        self.assertAsRevisionId(b'r1', '-2')
        self.assertAsRevisionId(b'r2', '-1')
        self.assertAsRevisionId(b'alt_r2', '1.1.1')

    def test_as_tree(self):
        tree = self.get_as_tree('0')
        self.assertEqual(_mod_revision.NULL_REVISION, tree.get_revision_id())
        tree = self.get_as_tree('1')
        self.assertEqual(b'r1', tree.get_revision_id())
        tree = self.get_as_tree('2')
        self.assertEqual(b'r2', tree.get_revision_id())
        tree = self.get_as_tree('-2')
        self.assertEqual(b'r1', tree.get_revision_id())
        tree = self.get_as_tree('-1')
        self.assertEqual(b'r2', tree.get_revision_id())
        tree = self.get_as_tree('1.1.1')
        self.assertEqual(b'alt_r2', tree.get_revision_id())