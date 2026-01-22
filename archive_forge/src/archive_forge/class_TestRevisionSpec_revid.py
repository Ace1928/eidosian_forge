import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
class TestRevisionSpec_revid(TestRevisionSpec):

    def test_in_history(self):
        self.assertInHistoryIs(1, b'r1', 'revid:r1')
        self.assertInHistoryIs(2, b'r2', 'revid:r2')

    def test_missing(self):
        self.assertInvalid('revid:r3', invalid_as_revision_id=False)

    def test_merged(self):
        """We can reach revisions in the ancestry"""
        self.assertInHistoryIs(None, b'alt_r2', 'revid:alt_r2')

    def test_not_here(self):
        self.tree2.commit('alt third', rev_id=b'alt_r3')
        self.assertInvalid('revid:alt_r3', invalid_as_revision_id=False)

    def test_in_repository(self):
        """We can get any revision id in the repository"""
        self.tree2.commit('alt third', rev_id=b'alt_r3')
        self.tree.branch.fetch(self.tree2.branch, b'alt_r3')
        self.assertInHistoryIs(None, b'alt_r3', 'revid:alt_r3')

    def test_unicode(self):
        """We correctly convert a unicode ui string to an encoded revid."""
        revision_id = '☃'.encode()
        self.tree.commit('unicode', rev_id=revision_id)
        self.assertInHistoryIs(3, revision_id, 'revid:☃')
        self.assertInHistoryIs(3, revision_id, 'revid:' + revision_id.decode('utf-8'))

    def test_as_revision_id(self):
        self.assertAsRevisionId(b'r1', 'revid:r1')
        self.assertAsRevisionId(b'r2', 'revid:r2')
        self.assertAsRevisionId(b'alt_r2', 'revid:alt_r2')