import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
class TestRevisionSpec_tag(TestRevisionSpec):

    def make_branch_and_tree(self, relpath):
        return TestRevisionSpec.make_branch_and_tree(self, relpath, format='dirstate-tags')

    def test_from_string_tag(self):
        spec = RevisionSpec.from_string('tag:bzr-0.14')
        self.assertIsInstance(spec, RevisionSpec_tag)
        self.assertEqual(spec.spec, 'bzr-0.14')

    def test_lookup_tag(self):
        self.tree.branch.tags.set_tag('bzr-0.14', b'r1')
        self.assertInHistoryIs(1, b'r1', 'tag:bzr-0.14')
        self.tree.branch.tags.set_tag('null_rev', b'null:')
        self.assertInHistoryIs(0, b'null:', 'tag:null_rev')

    def test_failed_lookup(self):
        self.assertRaises(errors.NoSuchTag, self.get_in_history, 'tag:some-random-tag')

    def test_as_revision_id(self):
        self.tree.branch.tags.set_tag('my-tag', b'r2')
        self.tree.branch.tags.set_tag('null_rev', b'null:')
        self.assertAsRevisionId(b'r2', 'tag:my-tag')
        self.assertAsRevisionId(b'null:', 'tag:null_rev')
        self.assertAsRevisionId(b'r1', 'before:tag:my-tag')