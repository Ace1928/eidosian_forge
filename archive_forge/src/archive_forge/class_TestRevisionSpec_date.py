import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
class TestRevisionSpec_date(TestRevisionSpec):

    def setUp(self):
        super(TestRevisionSpec, self).setUp()
        new_tree = self.make_branch_and_tree('new_tree')
        self.revid1 = new_tree.commit('Commit one', timestamp=time.time() - 60 * 60 * 24)
        self.revid2 = new_tree.commit('Commit two')
        self.revid3 = new_tree.commit('Commit three')
        self.tree = new_tree

    def test_tomorrow(self):
        self.assertInvalid('date:tomorrow')

    def test_today(self):
        self.assertInHistoryIs(2, self.revid2, 'date:today')
        self.assertInHistoryIs(1, self.revid1, 'before:date:today')

    def test_yesterday(self):
        self.assertInHistoryIs(1, self.revid1, 'date:yesterday')

    def test_invalid(self):
        self.assertInvalid('date:foobar', extra='\ninvalid date')
        self.assertInvalid('date:20040404', extra='\ninvalid date')
        self.assertInvalid('date:2004-4-4', extra='\ninvalid date')

    def test_day(self):
        now = datetime.datetime.now()
        self.assertInHistoryIs(2, self.revid2, 'date:%04d-%02d-%02d' % (now.year, now.month, now.day))

    def test_as_revision_id(self):
        self.assertAsRevisionId(self.revid2, 'date:today')