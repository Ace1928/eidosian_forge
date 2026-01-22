import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
class TestRevisionSpec(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/a'])
        self.tree.lock_write()
        self.addCleanup(self.tree.unlock)
        self.tree.add(['a'])
        self.tree.commit('a', rev_id=b'r1')
        self.tree2 = self.tree.controldir.sprout('tree2').open_workingtree()
        self.tree2.commit('alt', rev_id=b'alt_r2')
        self.tree.merge_from_branch(self.tree2.branch)
        self.tree.commit('second', rev_id=b'r2')

    def get_in_history(self, revision_spec):
        return spec_in_history(revision_spec, self.tree.branch)

    def assertInHistoryIs(self, exp_revno, exp_revision_id, revision_spec):
        rev_info = self.get_in_history(revision_spec)
        self.assertEqual(exp_revno, rev_info.revno, 'Revision spec: %r returned wrong revno: %r != %r' % (revision_spec, exp_revno, rev_info.revno))
        self.assertEqual(exp_revision_id, rev_info.rev_id, 'Revision spec: %r returned wrong revision id: %r != %r' % (revision_spec, exp_revision_id, rev_info.rev_id))

    def assertInvalid(self, revision_spec, extra='', invalid_as_revision_id=True):
        try:
            self.get_in_history(revision_spec)
        except InvalidRevisionSpec as e:
            self.assertEqual(revision_spec, e.spec)
            self.assertEqual(extra, e.extra)
        else:
            self.fail('Expected InvalidRevisionSpec to be raised for %r.in_history' % (revision_spec,))
        if invalid_as_revision_id:
            try:
                spec = RevisionSpec.from_string(revision_spec)
                spec.as_revision_id(self.tree.branch)
            except InvalidRevisionSpec as e:
                self.assertEqual(revision_spec, e.spec)
                self.assertEqual(extra, e.extra)
            else:
                self.fail('Expected InvalidRevisionSpec to be raised for %r.as_revision_id' % (revision_spec,))

    def assertAsRevisionId(self, revision_id, revision_spec):
        """Calling as_revision_id() should return the specified id."""
        spec = RevisionSpec.from_string(revision_spec)
        self.assertEqual(revision_id, spec.as_revision_id(self.tree.branch))

    def get_as_tree(self, revision_spec, tree=None):
        if tree is None:
            tree = self.tree
        spec = RevisionSpec.from_string(revision_spec)
        return spec.as_tree(tree.branch)