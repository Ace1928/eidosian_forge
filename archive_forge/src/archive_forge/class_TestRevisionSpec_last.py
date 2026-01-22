import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
class TestRevisionSpec_last(TestRevisionSpec):

    def test_positive(self):
        self.assertInHistoryIs(2, b'r2', 'last:1')
        self.assertInHistoryIs(1, b'r1', 'last:2')
        self.assertInHistoryIs(0, b'null:', 'last:3')

    def test_empty(self):
        self.assertInHistoryIs(2, b'r2', 'last:')

    def test_negative(self):
        self.assertInvalid('last:-1', extra='\nyou must supply a positive value')

    def test_missing(self):
        self.assertInvalid('last:4')

    def test_no_history(self):
        tree = self.make_branch_and_tree('tree3')
        self.assertRaises(errors.NoCommits, spec_in_history, 'last:', tree.branch)

    def test_not_a_number(self):
        last_e = None
        try:
            int('Y')
        except ValueError as e:
            last_e = e
        self.assertInvalid('last:Y', extra='\n' + str(last_e))

    def test_as_revision_id(self):
        self.assertAsRevisionId(b'r2', 'last:1')
        self.assertAsRevisionId(b'r1', 'last:2')