import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
class TestRevisionSpec_dwim(TestRevisionSpec):

    def test_dwim_spec_revno(self):
        self.assertInHistoryIs(2, b'r2', '2')
        self.assertAsRevisionId(b'alt_r2', '1.1.1')

    def test_dwim_spec_revid(self):
        self.assertInHistoryIs(2, b'r2', 'r2')

    def test_dwim_spec_tag(self):
        self.tree.branch.tags.set_tag('footag', b'r1')
        self.assertAsRevisionId(b'r1', 'footag')
        self.tree.branch.tags.delete_tag('footag')
        self.assertRaises(InvalidRevisionSpec, self.get_in_history, 'footag')

    def test_dwim_spec_tag_that_looks_like_revno(self):
        self.tree.branch.tags.set_tag('3', b'r2')
        self.assertAsRevisionId(b'r2', '3')
        self.build_tree(['tree/b'])
        self.tree.add(['b'])
        self.tree.commit('b', rev_id=b'r3')
        self.assertAsRevisionId(b'r3', '3')

    def test_dwim_spec_date(self):
        self.assertAsRevisionId(b'r1', 'today')

    def test_dwim_spec_branch(self):
        self.assertInHistoryIs(None, b'alt_r2', 'tree2')

    def test_dwim_spec_nonexistent(self):
        self.assertInvalid('somethingrandom', invalid_as_revision_id=False)
        self.assertInvalid('-1.1', invalid_as_revision_id=False)
        self.assertInvalid('.1', invalid_as_revision_id=False)
        self.assertInvalid('1..1', invalid_as_revision_id=False)
        self.assertInvalid('1.2..1', invalid_as_revision_id=False)
        self.assertInvalid('1.', invalid_as_revision_id=False)

    def test_append_dwim_revspec(self):
        original_dwim_revspecs = list(RevisionSpec_dwim._possible_revspecs)

        def reset_dwim_revspecs():
            RevisionSpec_dwim._possible_revspecs = original_dwim_revspecs
        self.addCleanup(reset_dwim_revspecs)
        RevisionSpec_dwim.append_possible_revspec(RevisionSpec_bork)
        self.assertAsRevisionId(b'r1', 'bork')

    def test_append_lazy_dwim_revspec(self):
        original_dwim_revspecs = list(RevisionSpec_dwim._possible_revspecs)

        def reset_dwim_revspecs():
            RevisionSpec_dwim._possible_revspecs = original_dwim_revspecs
        self.addCleanup(reset_dwim_revspecs)
        RevisionSpec_dwim.append_possible_lazy_revspec('breezy.tests.test_revisionspec', 'RevisionSpec_bork')
        self.assertAsRevisionId(b'r1', 'bork')